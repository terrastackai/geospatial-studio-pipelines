# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


"""
This component reads the output from preprocessing and puts together a request to the inference service.
"""

# Dependencies
# pip install rioxarray shapely geopandas numpy==2.3.2 xarray opentelemetry-distro opentelemetry-exporter-otlp buildingregulariser matplotlib

import os
import json
import rioxarray
from gfm_data_processing.common import logger, notify_gfmaas_ui, report_exception
from gfm_data_processing.metrics import MetricManager
from postprocess_generic_helper_functions import (
    make_rgb,
    mask_ocean,
    resize_image,
    mask_from_url,
    geojson_to_tiff,
    mask_and_set_to,
    zip_inference_data,
    read_json_with_retries,
    get_lulc_tile_for_input,
    regularize_by_technique,
)

# Uncomment next 2 lines for local testing
# import dotenv
# dotenv.load_dotenv()

# inference folder
inference_folder = os.environ.get("inference_folder", "")

# inference_id
inference_id = os.environ.get("inference_id", "test-inference-1")

# task_id
task_id = os.environ.get("task_id", f"{inference_id}-task_0")

process_id = os.getenv("process_id", "postprocess-generic")

metric_manager = MetricManager(component_name=process_id)


def post_process_original(inference_dict, imputed_input_image, raw_input_image, xds, i=0):
    """Post processing the output file by Masking on output image

    Parameters
    ----------
    inference_dict : dict
        inference def dictionary
    imputed_input_image : str
        imputed_input_image file_path
    raw_input_image : str
        original input image file_path
    xds : xarray
        model_output_image xarray
    i : int, optional
        index for imputed_input_image, by default 0
    """
    ids = rioxarray.open_rasterio(imputed_input_image)

    masked = False

    if "cloud_masking" in inference_dict["post_processing"]:
        if inference_dict["post_processing"]["cloud_masking"] == "True":
            logger.info("*********** Cloud masking ***********")

            masking_type = "cloud_masking"  # As set in config file.
            xds = mask_and_set_to(xds, ids, masking_type, inference_dict, i)
            masked = True

    if "snow_ice_masking" in inference_dict["post_processing"]:
        if inference_dict["post_processing"]["snow_ice_masking"] == "True":
            logger.info("*********** Snow and ice masking ***********")

            masking_type = "snow_ice_masking"
            xds = mask_and_set_to(xds, ids, masking_type, inference_dict, i)
            masked = True

    if "permanent_water_masking" in inference_dict["post_processing"]:
        if inference_dict["post_processing"]["permanent_water_masking"] == "True":
            logger.info("*********** Permanent water masking ***********")

            masking_type = "permanent_water_masking"
            if "permanent_water_masking" in inference_dict["data_connector_config"][i]:
                encoding_type = inference_dict["data_connector_config"][i][masking_type]["encoding"]
                if encoding_type == "sentinel2_lulc":
                    try:
                        lulc_file_path = get_lulc_tile_for_input(raw_input_image)
                        ids = rioxarray.open_rasterio(lulc_file_path)
                    except Exception as e:
                        logger.exception(e)
                        logger.error("Issue accessing LULC tiles. Fall back to using scene classification layer.")
                else:
                    inference_dict["data_connector_config"][i][masking_type]["encoding"] = "sentinel2_scl"
                    inference_dict["data_connector_config"][i][masking_type]["band"] = "SCL"
                # Create mask for permanent water
                xds = mask_and_set_to(xds, ids, masking_type, inference_dict, i)
                masked = True
            else:
                logger.error(
                    "*********** Error running post processing: permanent_water_masking post_processing key missing from inference dictionary ***********"
                )

    if "ocean_masking" in inference_dict["post_processing"]:
        if inference_dict["post_processing"]["ocean_masking"] == "True":
            logger.info("*********** Ocean masking ***********")
            xds = mask_ocean(xds)
            masked = True

    if "mask_from_url" in inference_dict["post_processing"]:
        if inference_dict["post_processing"].get("mask_from_url") not in ["", "None", "False", None, {}]:
            # Use the location for the mask geojson to task
            logger.info("*********** Custom user-defined masking ***********")
            custom_mask_location = inference_dict["post_processing"]["mask_from_url"].get(
                "inference_folder_mask_location"
            ) or inference_dict["post_processing"]["mask_from_url"].get("url")
            buffer_size = float(inference_dict["post_processing"]["mask_from_url"].get("buffer_size_m", 100.0))
            xds = mask_from_url(custom_mask_location, xds, buffer_size)
            masked = True

    return masked, xds


def regularize_prediction(inference_dict, raster_to_regularize):
    """Regularize the post processed prediction raster file or raw prediction is masking is off

    Parameters
    ----------
    inference_dict : dict
        inference def dictionary
    raster_to_regularize : str
        can be prediction or masked prediction file_path

    Returns
    -------
    Union[gpd.GeoDataFrame, List[Polygon]
        GeoDataFrame or list of shapely Polygons with regularized building footprints
    """
    regularized = False
    regularized_vector = None

    if "regularization" in inference_dict["post_processing"] and raster_to_regularize:
        if inference_dict["post_processing"].get("regularization") == "True":
            # TODO Allow user to select which function to use. and which args to pass. For now use the adaptive regularization technique
            regularized_vector = regularize_by_technique(
                raster_to_regularize=raster_to_regularize,
                technique="adaptive_regularization",
                attribute_name="buildings",
                simplify_tolerance=0,
                area_threshold=0.0000001,
                preserve_shape=True,
            )
            regularized = True

    return regularized, regularized_vector


def update_processed_paths(post_process_outputs, active_post_processing_steps, task_dict):
    """Function to update task_dict with the new file outputs

    Parameters
    ----------
    post_process_outputs : dict
        Outputs of the post processing steps
    active_post_processing_steps : List(Dict(str, All))
        List of all active post processing steps
    task_dict : dict
        Dict with the tasks details written

    Returns
    -------
    dict
        Updated task dict
    """

    for registered in active_post_processing_steps:
        print(registered['name'])

        # grab function name
        func_name = registered['name']

        # index outputs with func name
        print(post_process_outputs[func_name])


        # append func_name with model_output_<>_image
        post_processed_model_output = f"model_output_{func_name}_image"

        # assign this name to the processed_file_path file_path
        task_dict[post_processed_model_output] = post_process_outputs[func_name]['processed_file_path']

    return task_dict


def update_input_paths(active_post_processing_steps, model_output_image_path):
    """Function to populate input paths


    Parameters
    ----------
    active_post_processing_steps : List[Dict[str, Any]
        list of dicts of activated post processing steps
    model_output_image_path : str
        Model output image path

    Returns
    -------
    dict
        Post processing steps updated with img_path and out_path
    """
    if active_post_processing_steps:
        for step in active_post_processing_steps:
            # grab input image & file extension
            img_path_root, img_path_ext = os.path.splitext(model_output_image_path)

            # grab geoserver file extension from function name i.e im2poly_regularize
            output_file_ext = step.get('params',{}).get("geoserver_suffix_extension",{})
 
            # grab geoserver file suffix from function name i.e im2poly_regularize
            suffix = step['name']

            if suffix:
                # Generate the input and output file paths
                # If file_extension not provided, save as a ".tif"
                if output_file_ext:
                    model_output_post_processed_image = model_output_image_path.replace(img_path_ext, f"_{suffix}.{output_file_ext}")
                else:
                    output_file_ext = "tif"
                    model_output_post_processed_image = model_output_image_path.replace(img_path_ext, f"_{suffix}.{output_file_ext}")

                # Update the active_post_processing_steps dict with img_paths and out_paths
                step['params']['img_path'] = model_output_image_path
                step['params']['out_path'] = model_output_post_processed_image
            else:
                logger.error("Geoserver suffix expected to create the new file path. ")


        return active_post_processing_steps

@metric_manager.count_failures(inference_id=inference_id, task_id=task_id)
@metric_manager.record_duration(inference_id=inference_id, task_id=task_id)
def postprocess_generic_single():
    # Optional: notify UI that postprocessing is starting
    try:
        notify_gfmaas_ui(
            event_id=inference_id,
            task_id=task_id,
            event_status="Post-processing results ..",
        )
        ######################################################################################################
        ###  Parse the inference and task configs from file
        ######################################################################################################

        inference_config_path = f"{inference_folder}/{inference_id}_config.json"
        with open(inference_config_path, "r") as fp:
            inference_dict = json.load(fp)

        task_folder = f"{inference_folder}/{task_id}"

        task_config_path = f"{task_folder}/{task_id}_config.json"
        task_dict = read_json_with_retries(filepath=task_config_path, max_retries=5)

        model_input_original_image = task_dict["original_input_image"]
        model_input_imputed_image = task_dict["imputed_input_image"]
        model_output_image = task_dict["model_output_image"]

        ######################################################################################################
        ###  Resize the output image
        ######################################################################################################
        logger.info("*********** Resizing output image ***********")
        resize_image(model_output_image, input_image=model_input_imputed_image)

        ######################################################################################################
        ###  Input image RGB
        ######################################################################################################
        logger.info("*********** Making input RGB ***********")
        rgb_filename, rgb_index = make_rgb(model_input_original_image, inference_dict)
        task_dict["model_input_original_image_rgb"] = rgb_filename

        ######################################################################################################
        ###  Masking on output image
        ######################################################################################################
        logger.info("*********** Optional masking steps ***********")

        xds = rioxarray.open_rasterio(model_output_image)
        xds.rio.write_nodata(-9999, inplace=True)
        masked = False
        regularized = False
        masked_output_path = None
        post_processing_config = inference_dict.get("post_processing",{})

        # Open original image if any post processing is taking place
        if "post_processing" in inference_dict.keys():
            if isinstance(model_input_imputed_image, list):
                masked, xds = post_process_original(
                    inference_dict=inference_dict,
                    imputed_input_image=model_input_imputed_image[rgb_index],
                    raw_input_image=model_input_original_image[rgb_index],
                    xds=xds,
                    i=rgb_index,
                )
            else:
                masked, xds = post_process_original(
                    inference_dict=inference_dict,
                    imputed_input_image=model_input_imputed_image,
                    raw_input_image=model_input_original_image,
                    xds=xds,
                    i=rgb_index,
                )
        else:
            logger.error(
                "*********** Error running post processing: post_processing key missing from inference dictionary ***********"
            )

        if masked:
            xds.rio.write_nodata(-9999, inplace=True)
            masked_output_path = model_output_image.replace(".tif", "_masked.tif")
            xds.rio.to_raster(masked_output_path)
            task_dict["model_output_image_masked"] = masked_output_path

        raster_to_regularize = masked_output_path if masked_output_path else model_output_image
        regularized, regularized_vector = regularize_prediction(
            inference_dict=inference_dict, raster_to_regularize=raster_to_regularize
        )

        if regularized:
            # TODO: Create a function to create this on the fly.
            model_output_regularized_image = model_output_image.replace(".tif", "_adaptive_regularized.tif")
            # Convert the geojson to a tif file
            geojson_to_tiff(
                prediction_tif_file_path=model_output_image,
                geodataframe=regularized_vector,
                output_tif_file_path=model_output_regularized_image,
                attribute_name="buildings",
            )
            task_dict["model_output_regularized_image"] = model_output_regularized_image

        logger.info("*********** Optional post_processing steps ***********")

        # Run all the registered post_procesing steps
        from post_process.post_process import post_process
        active_postprocessing_steps = post_processing_config.get("regularization_custom",{})
        logger.debug(
            f"*********Active Postprocessing steps:********** {active_postprocessing_steps}"
        )

        if active_postprocessing_steps:
            # TODO: For now, we download the python script at this point and register the function.

            download_plugins_config = post_processing_config.get("download_scripts",{})
            download_scripts = download_plugins_config.get("activated", "False")
            logger.info(f"Download custom post processing scripts activated: {download_scripts}")
            if download_scripts == "True":
                from post_process.post_process.sdk import download_plugins_to_local

                logger.info("*********** Downloading custom post processing scripts ***********")

                results, saved_plugins = download_plugins_to_local(
                    verify_ssl=download_plugins_config.get("verify_ssl", "False"),
                    tmp_plugins_dir=task_folder,
                    plugins_list=download_plugins_config.get(
                        "plugins_list",
                        [
                            {
                                "url": "https://s3.us-east.cloud-object-storage.appdomain.cloud/geospatial-studio-example-data/example_post_processing_scripts/user_masking_testing.py?response-content-type=application%2Foctet-stream&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=1c6317aac0054b9890797f09f217b54e%2F20251202%2Fus-east%2Fs3%2Faws4_request&X-Amz-Date=20251202T124637Z&X-Amz-Expires=7200&X-Amz-SignedHeaders=host&X-Amz-Signature=8a7a4cf870d4bb517b2769460206a5741ef0c0f3d10fe336dfa4ff586e596b5a",
                                "filename": "user_masking_testing.py",
                            },
                            {
                                "url": "https://s3.us-east.cloud-object-storage.appdomain.cloud/geospatial-studio-example-data/example_post_processing_scripts/cloud_masking_testing.py?response-content-type=application%2Foctet-stream&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=1c6317aac0054b9890797f09f217b54e%2F20251202%2Fus-east%2Fs3%2Faws4_request&X-Amz-Date=20251202T130502Z&X-Amz-Expires=7200&X-Amz-SignedHeaders=host&X-Amz-Signature=08ad706c6cdea35b004593aa7c247ba6b503421d618b10db340fd521d4c06632",
                                "filename": "cloud_masking_testing.py",
                            },
                        ],
                    ),
                )
                logger.info(f"Downloaded plugins: {saved_plugins}")
            # Update paths to the ones in the cluster/env
            active_postprocessing_steps_updated_input_paths = update_input_paths(
                active_post_processing_steps=active_postprocessing_steps,
                model_output_image_path=model_output_image,
            )
            logger.info(f"Updated paths: {active_postprocessing_steps_updated_input_paths}")

            # Run all the registered steps.
            post_process_outputs = post_process(
                raster_to_regularize,
                steps_config=active_postprocessing_steps_updated_input_paths,
                plugins_dir="post_process/post_process/generic/",
            )
            # Write to inference dict
            task_dict_updated = update_processed_paths(
                post_process_outputs=post_process_outputs,
                active_post_processing_steps=active_postprocessing_steps_updated_input_paths,
                task_dict=task_dict
            )
            logger.info(
                f"********* Updated task_dict Postprocessing steps:********** {task_dict_updated}"
            )

            logger.info(f"Post_process outputs: {post_process_outputs}")

        # TODO: Move to Geoserver
        # - Is it possible for geoserver to know steps automatically and push the layers?

        zip_inference_data(task_folder)
        if active_postprocessing_steps:
            with open(task_config_path, "w") as fp:
                json.dump(task_dict, fp, indent=4)
        else:
            with open(task_config_path, "w") as fp:
                json.dump(task_dict, fp, indent=4)
    except Exception as ex:
        report_exception(
            event_id=inference_id,
            task_id=task_id,
            error_code="1043",
            message=f"Postprocessing failed with: {ex}",
            event_detail_type="Inf:Task:Failed",
            verbose=True,  # set to False if you want less detail
            raise_exception=False,
        )
        raise  # Remove this line if you want to continue after error, else it will stop on error.
    finally:
        logger.info(f"{inference_id}: ********* PostProcessing Complete **********")


if __name__ == "__main__":
    postprocess_generic_single()
