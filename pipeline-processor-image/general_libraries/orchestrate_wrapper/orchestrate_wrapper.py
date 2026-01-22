# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


import os
import time
import json
import yaml 
import random
import logging
import requests
import subprocess
import contextlib
from pathlib import Path
from datetime import datetime, timezone
from urllib.parse import urljoin
from sqlalchemy import create_engine, text

# Uncomment next 2 lines for local testing
import dotenv

dotenv.load_dotenv()

process_id = os.getenv("process_id", "sentinelhub_connector")
process_exec = os.getenv("process_exec", "python sentinelhub_connector_single.py")
orchestrate_db_uri = os.getenv("orchestrate_db_uri", "")
inf_task_table = os.getenv("inference_task_table", "task")
stop_exit_code = int(os.getenv("stop_exit_code", 9876))
gfmaas_api_base_url = os.getenv("gfmaas_api_base_url", "")
gfmaas_api_key = os.getenv("gfmaas_api_key", "")
log_level = os.getenv("log_level", "INFO")

# import


def configure_logger(log_level):
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)

    handler = logging.StreamHandler()
    handler.setLevel(log_level)
    # Create a formatter to specify the format of the log messages
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s"
    )
    handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(handler)

    return logger


def notify_gfmaas_ui(
    event_id: str,
    task_id: str = None,
    event_status: str = None,
    event_detail_type: str = None,
    event_processed_result: dict = None,
):
    """Helper method to notify Gfmaas-UI
    Parameters
    ----------
    event_id : str
        The event_id sent from the client when the inference was started.
    task_id : str
        The task_id for specific task related to inference; an inference is segmented to multiple tasks based on bboxes/dates.
    event_status : str
        The current status of the event with `event_id` in the inference server
    event_detail_type : str
        The event detail type of the event with `event_id` in the inference server
    event_processed_result : dict
        The dict of the result for the event with `event_id` in the inference server
    """
    gfmaas_webhooks_url = urljoin(gfmaas_api_base_url, "notifications")
    gfmaas_webhooks_headers = {
        "Content-Type": "application/json",
        "X-API-KEY": gfmaas_api_key,
    }

    event_data = {
        "event_id": event_id,
        "detail_type": event_detail_type or "Inf:Task:Notify",
        "source": "com.inference-v2-pipelines-service.ibm",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "detail": event_processed_result or {"message": event_status, "task_id": task_id},
    }

    # try-except to capture a failed connection attempt, we want to continue with the inference regardless...
    try:
        response = requests.post(
            url=gfmaas_webhooks_url,
            headers=gfmaas_webhooks_headers,
            json=event_data,
            timeout=29,
            verify=False,
        )
        if response.status_code not in (200, 201):
            logger.error(
                "Failed to send task status. Reason: (%s)> %s",
                response.status_code,
                response.text,
            )
        else:
            logger.info("Sent a notification to calling service")
    except Exception as ex:
        logger.error("Failed to send task status. Reason: (%s)", ex)


def run_and_log(task_id, process_exec, process_id, inference_folder):
    std_out_log_name = f"{inference_folder}/{task_id}/{task_id}-{process_id}-stdout.log"
    std_err_log_name = f"{inference_folder}/{task_id}/{task_id}-{process_id}-stderr.log"
    try:
        with open(std_out_log_name, "w") as so:
            with open(std_err_log_name, "w") as se:
                with contextlib.redirect_stdout(so):
                    with contextlib.redirect_stderr(se):
                        ("-----INVOKING TASK-----------------------------------")
                        logger.debug("-----INVOKING TASK-----------------------------------")
                        print("-----INVOKING TASK-----------------------------------")
                        logger.debug(f"Task ID: {task_id}")
                        print(f"Task ID: {task_id}")
                        logger.debug(f"Command: {process_exec}")
                        print(f"Command: {process_exec}")
                        try:
                            result = subprocess.run(
                                process_exec,
                                shell=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,
                                check=True,
                                env=os.environ.copy(),
                            )
                            output = result.stdout.decode("utf-8")
                            logger.debug(f"Output: {str(output)}")
                            print(f"Output: {str(output)}")
                            logger.debug(f"Return code: {result.returncode}")
                            print(f"Return code: {result.returncode}")
                            return result.returncode
                        except subprocess.CalledProcessError as sub_ex:
                            logger.error(f"Task ID: {task_id} Command: {process_exec} exited with error: {sub_ex}")
                            if sub_ex.stdout:
                                error_stdout = sub_ex.stdout.decode("utf-8")
                                logger.debug(f"Error stdout: {error_stdout}")
                                print(f"Error stdout: {error_stdout}")
                            if sub_ex.stderr:
                                error_stderr = sub_ex.stderr.decode("utf-8")
                                logger.debug(f"Error stderr: {error_stderr}")
                                print(f"Error stderr: {error_stderr}")
                            return sub_ex.returncode
                        except Exception as ex:
                            logger.error(f"Task ID: {task_id} Command: {process_exec} exited with error: {ex}")
                            return 500
    except Exception as ex:
        logger.error(f"Task ID: {task_id} Command: {process_exec} exited with error: {ex}")
        return 500


def grab_new_task(engine, process_id):
    query = None
    with engine.connect() as conn:
        start_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

        # TO DO: add explanantion here
        task_search_sql = text(
            f"""UPDATE {inf_task_table} SET pipeline_steps = jsonb_set(jsonb_set(pipeline_steps, array[elem_index::text, 'status'], '"RUNNING"'::jsonb), array[elem_index::text, 'start_time'], '"{start_time}"'::jsonb)
        FROM (
            select 
                pos- 1 as elem_index, id as tid
            FROM {inf_task_table} t CROSS JOIN LATERAL jsonb_array_elements(t.pipeline_steps) AS p(j),
                jsonb_array_elements(pipeline_steps) with ordinality arr(elem, pos)
            where
                elem->>'process_id' = '{process_id}' AND p->>'process_id' = '{process_id}' AND p->>'status'='READY'
            ORDER BY priority DESC, id ASC LIMIT 1 FOR UPDATE SKIP LOCKED) AS sub_arrange
            WHERE id=tid RETURNING task_id, inference_id, inference_folder, status;"""
        )

        # print(task_search_sql)
        query = conn.execute(task_search_sql).fetchall()
        conn.commit()

        if len(query) > 0:
            query = query[0]
            task_id = query[0]
            task_status = query[3]
            logger.info(f">>>>>>>> Grabbed new task: {task_id}")

            if task_status == "READY":
                update_overall_status = text(
                    f"""UPDATE {inf_task_table} SET status = 'RUNNING'  WHERE task_id = '{task_id}';"""
                )
                conn.execute(update_overall_status)
                conn.commit()


            logger.info(f">>>>>>>> Returning query info: {query}")
    return query


def update_status_after_run(engine, process_id, inference_id, task_id, new_state):
    task_terminal_status_set = False
    end_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    query = None
    with engine.connect() as conn:
        task_pipeline_sql = text(f"SELECT pipeline_steps FROM {inf_task_table} WHERE task_id = '{task_id}';")
        # print(task_pipeline_sql)
        pipeline_steps = conn.execute(task_pipeline_sql).fetchall()[0][0]

    for s in range(0, len(pipeline_steps)):
        if pipeline_steps[s]["process_id"] == process_id:
            pipeline_steps[s]["status"] = new_state
            # pipeline_steps[s]['start_time'] = start_time
            pipeline_steps[s]["end_time"] = end_time
            next_step = pipeline_steps[s]["step_number"] + 1

    if new_state == "FINISHED":
        if next_step < len(pipeline_steps):
            for s in range(0, len(pipeline_steps)):
                if pipeline_steps[s]["step_number"] == next_step:
                    pipeline_steps[s]["status"] = "READY"

    if new_state == "STOPPED":
        if next_step < len(pipeline_steps):
            for s in range(0, len(pipeline_steps)):
                if pipeline_steps[s]["process_id"] == process_id:
                    pipeline_steps[s]["status"] = "FINISHED"
                elif pipeline_steps[s]["step_number"] >= next_step:
                    pipeline_steps[s]["status"] = "STOPPED"

    with engine.connect() as conn:
        update_status = text(
            f"""UPDATE {inf_task_table} SET pipeline_steps = '{json.dumps(pipeline_steps)}'  WHERE task_id = '{task_id}';"""
        )
        # print(update_status)
        conn.execute(update_status)
        conn.commit()

        # check overall status
        if all([X["status"] == "FINISHED" for X in pipeline_steps]) == True:
            update_overall_status = text(
                f"""UPDATE {inf_task_table} SET status = 'FINISHED'  WHERE task_id = '{task_id}';"""
            )
            conn.execute(update_overall_status)
            conn.commit()
            task_terminal_status_set = True
        if any([X["status"] == "STOPPED" for X in pipeline_steps]) == True:
            update_overall_status = text(
                f"""UPDATE {inf_task_table} SET status = 'STOPPED'  WHERE task_id = '{task_id}';"""
            )
            conn.execute(update_overall_status)
            conn.commit()
            task_terminal_status_set = True
        if any([X["status"] == "FAILED" for X in pipeline_steps]) == True:
            update_overall_status = text(
                f"""UPDATE {inf_task_table} SET status = 'FAILED'  WHERE task_id = '{task_id}';"""
            )
            conn.execute(update_overall_status)
            conn.commit()
            task_terminal_status_set = True

    if task_terminal_status_set:
        notify_gfmaas_ui(
            event_id=inference_id,
            event_detail_type="Inf:Task:Updated",
            event_processed_result={
                "event_id": inference_id,
                "task_id": task_id,
            },
        )

def get_generic_processor_values(inference_folder: str, task_id: str):
    logger.info(f">>>>>> Detected generic-python-processor, about to read script from inference_config file")
    # read the script from the inference_config.yaml file
    inference_config_path = f"{inference_folder}/{inference_id}_config.json"
    with open(inference_config_path, "r") as fp:
        inference_dict = json.load(fp)

    # Get all the generic python processor config
    python_generic_processor_config = inference_dict.get('generic_processor', None)
    if python_generic_processor_config is None:
        logger.error(f">>>>>> No generic_processor found in inference_config.yaml for task {task_id}, exiting with error")
        update_status_after_run(engine, process_id, inference_id, task_id, "FAILED")

        return None

    name, status, description, processor_file_path, processor_parameters = (
            python_generic_processor_config.get(k, d)
            for k, d in [
                ("name", None),
                ("status", None),
                ("description", None),
                ("processor_file_path", None),
                ("processor_parameters", {}),
            ]
        )

    # if the status is failed/pending, raise error to warn user that the script is not uploaded to COS
    if status not in ["FINISHED"]:
        logger.error(
            f">>>>>> generic_processor script is not uploaded to storage for task {task_id} with status {status}. Kindly upload the script."
        )
        update_status_after_run(engine, process_id, inference_id, task_id, "FAILED")

        return None
    # if the status is finished, proceed to copy the script to the task folder
    if processor_file_path:
        bucket_path = Path(f"/data/{processor_file_path}")
        dest_path = Path(f"{inference_folder}/{task_id}/{processor_file_path}")
        if not bucket_path.is_file:
            os.copy(bucket_path, dest_path)
        else:
            logger.error(f">>>>>> generic_processor script file not found in storage for task {task_id} at path {bucket_path}.")
            update_status_after_run(engine, process_id, inference_id, task_id, "FAILED")

    
    return name, status, dest_path, processor_parameters

######################################################################################################
###  Main script
######################################################################################################

logger = configure_logger(log_level)
engine = create_engine(orchestrate_db_uri)
print(process_id)

while True:
    ######################################################################################################
    ###  Check for a new task
    ######################################################################################################

    data = grab_new_task(engine, process_id)
    # Data is a query from the database with (task_id, inference_id, inference_folder)
    print(data)

    ######################################################################################################
    ###  if a new task is found, run the process script
    ######################################################################################################
    if len(data) > 0:
        logger.info(f">>>>>> Query return: {data}")
        task_id = data[0]
        inference_id = str(data[1])
        inference_folder = data[2]

        os.environ["inference_id"] = inference_id
        os.environ["inference_folder"] = inference_folder
        os.environ["task_id"] = task_id

        # start_time = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')

        # if we want to run a generic python script, pull the correct
        if process_id=='generic-python-processor':

            processor_file_name, status, processor_file_path, processor_parameters = (
                get_generic_processor_values()
            )

            # then, get the script_params , and grab the process_exec from the parameters
            if processor_parameters:
                # finally, run the process_exec
                process_exec = f"python {inference_folder}/{task_id}/{processor_file_path}"
                for param_key, param_value in processor_parameters.items():
                    process_exec += f" --{param_key} {param_value}"
                logger.info(f">>>>>> Constructed process_exec for generic-python-processor: {process_exec}")
                return_value = run_and_log(task_id, process_exec, process_id, inference_folder)
                logger.info(f">>>>>> Return code: {return_value}")

            else:
                logger.error(f">>>>>> No processor_parameters found in inference_config.yaml for task {task_id}, exiting with error")
                update_status_after_run(engine, process_id, inference_id, task_id, "FAILED")
                continue

        #     process_exec=...
        #     # TO DO: read requirements from script header and pip install
        # cd into inference folder - get from inference_config
        # make a change in the gateway to make a change to the POST/inference to inference_config
        #
        # Alter sql syntax to return pipeline_steps :Not necessary

        # Here actually run the process code and capture the logs
        return_value = run_and_log(task_id, process_exec, process_id, inference_folder)
        logger.info(f">>>>>> Return code: {return_value}")

        if return_value == 0:
            logger.info(f">>>>>> Finished running code for {task_id}, about to update the status db")
            update_status_after_run(engine, process_id, inference_id, task_id, "FINISHED")

        elif return_value == stop_exit_code:
            logger.info(f">>>>>> Stop running code for {task_id}, about to update the status db")
            update_status_after_run(engine, process_id, inference_id, task_id, "STOPPED")

        else:
            logger.info(f">>>>>> Failed running code for {task_id}, about to update the status db")
            update_status_after_run(engine, process_id, inference_id, task_id, "FAILED")

    else:
        logger.info(
            f"------ {datetime.now().strftime('%Y-%m-%dT%H:%M:%S')} - No tasks found, waiting 10 seconds ---------"
        )
        time.sleep(8.0 + random.uniform(0.0, 4.0))
