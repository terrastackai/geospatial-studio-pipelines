# postprocess-generic-single component

## Installing pre-requisites for building the generic component

- Claimed. The component is built using claimed compiler. To install claimed [see docs](https://github.com/romeokienzler/c3?tab=readme-ov-file#install)

- Docker. Used to generate the docker images.

---

---

### How to build component

1. Copy gfm_logger, gfm_data_processing and orchestrate_wrapper to current directory where `postprocess-generic-single.py` lives

```
cp -r ../../general_libraries/gfm_logger . && cp -r ../../general_libraries/gfm_data_processing . && cp -r ../../general_libraries/orchestrate_wrapper .

```

2. Create component using c3; uses current directory `pwd` as the docker context; ensure main script and additional files/dirs are in the same folder

```
c3_create_operator --repository us.icr.io/gfmaas --dockerfile_template_path Dockerfile.template --log_level DEBUG --version v1.0.14-dec04-test0 --local_mode postprocess_generic_single.py postprocess_generic_helper_functions.py postprocess_regularization.py gfm_logger gfm_data_processing post_process
```

3. Remove gfm_logger and gfm_data_processing from current directory

```
rm -r gfm_logger gfm_data_processing orchestrate_wrapper ./*.cwl ./*.job.yaml postprocess_generic_single.yaml
```

---

---

### Adding a new post_processing step

You can add a new post-processing step in the pipelines to run extra steps for the generated model predictions.

For this, you are expected to have:

1. The python script that runs the post-processing step.
2. The script has a entry_point function that takes input as a raster(.tif , .nc) and outputs a raster(.nc,.tif) or a vector(.gpkg, .shp) that can be pushed to geoserver later on for visualization and a file_path to the file. The expected format output of the function:
```py
    {
        "processed_file": regularized_mask, # The post processed mask
        "processed_file_path": regularized_mask_path, # Path to the post processed mask
    }

```
`(regularized_mask, regularized_mask_path )`

Steps to integrate it in the pipelines.

1. Add the post-processing python script to the root of the `post_process/generic/` folder.
2. In your script, add the decorator `@register_step("<unique_name>")` to register your entry_point function to the post_processing steps.

With this, your post_processing script is now registered and can run with the default parameters.

To pass specific parameters to the defined functions:

1. Edit the inference_dict with specific parameters as a dictionary in the `regularization_custom` key.

```py
{
"post_processing": {
    "regularization": "True",
    "regularization_custom": {
        [
            {
                "name": "im2poly_regularize",
                "params": {
                    "geoserver_suffix_extension": "gpkg", #  file extension of file to be pushed to geoserver. If empty the file is .tif
                    "params": {
                        "simplify_tolerance": 2,
                        "parallel_threshold": 2.0,
                        "allow_45_degree": True,
                    },
                },
            },
            {
                "name": "example_post_processing",
                "params": {
                    "img_path": "imagePath" # TODO: Provide dynamically to codebase
                    "nodata_value" : -9999
                }
            }
        ]
    },
    }
}

```
2. Now the registered steps should run with the params provided and a file generated in the folder.


### Pushing the file to geoserver
1. Create a new layer to push the post-processed file.

```py
  "geoserver_push": [
    {
      "z_index": 1,
      "workspace": "geofm",
      "layer_name": "<name of the layer>",
      "file_suffix": "<file_suffix>",
      "display_name": "<Display name?",
      "filepath_key": "<file_path_provided_in_post_process_step>",
      "geoserver_style": {
        # Replace this style with appropriate style for the generated file
        "segmentation": [
          {
            "color": "#78262eff",
            "label": "no-buildings",
            "opacity": 0,
            "quantity": "0"
          },
          {
            "color": "#3b0f8cff",
            "label": "buildings",
            "opacity": 1,
            "quantity": "1"
          },
        ]
      },
      "visible_by_default": "True"
    }
  ]
```

### Notebook example
[Notebook with examples](docs/notebooks/registering_new_post_processing_step.ipynb)

### Further Works

This component can be adapted for other modalities e.g. sentinelhub, openeo, etc. Just needs to replace the initial part of fetching the data.
