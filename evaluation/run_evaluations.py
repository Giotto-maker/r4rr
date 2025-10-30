import os
import torch
import papermill as pm
from concurrent.futures import ThreadPoolExecutor, as_completed

input_notebook = "evaluate.ipynb"

def execute_notebook(notebook_path, output_path, parameters):
    """Function to execute a notebook with given parameters."""
    try:
        pm.execute_notebook(
            notebook_path,
            output_path,
            parameters=parameters
        )
        print(f"Execution completed for: {output_path}")
    except Exception as e:
        print(f"Error executing {output_path}: {e}")


def run_parallel_executions(parameter_sets):
    max_workers = len(parameter_sets)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i, params in enumerate(parameter_sets):
            output_path = os.path.join("papermill", f"{i+1}.ipynb")
            futures.append(executor.submit(execute_notebook, input_notebook, output_path, params))

        # Wait for all futures to complete and print their results
        for future in as_completed(futures):
            result = future.result()  # This will raise exceptions if any occurred
    
    # Clean up CUDA memory
    torch.cuda.empty_cache()


if __name__ == "__main__":

    '''
    hide_shapes=[
        [0],
        [],
        [1],
        [2],
        [],
        [0,1],   
        [1],     
        [2],     
        [],      
        [0],   
        [0,1],   
        [1],     
        [0],     
        [2],     
        [0,1],
        [0,1], 
        [0,2],   
        [0,1],  
        [1,2],   
        [0,1],  
    ]

    hide_colors=(
        [],
        [1],
        [],
        [],
        [0],
        [],
        [1],
        [0],
        [1,2],
        [0],
        [0],
        [0,2],
        [0,1],
        [0,2],
        [2],
        [1,2],    
        [0,1],    
        [0,1],    
        [1,2],   
        [0,2],    
    )
    '''

    '''
    hide_lists=[
        [6],
        [2],
        [0],
        [3],
        [9],
        [3,6],
        [0,2],
        [1,7],
        [5,9],
        [4,8],
        [2,6,8],
        [0,3,8],
        [0,5,9],
        [1,4,7],
        [2,5,7],
        [0,1,4,7],
        [0,3,5,8],
        [3,5,6,7],
        [0,2,4,8],
        [2,5,6,9],
        [1,2,3,4,5],
        [2,3,6,8,9],
        [0,2,4,6,8],
        [0,5,6,7,8],
        [1,3,5,7,9],
        [4,5,6,7,8,9],
        [0,2,3,5,7,8],
        [0,1,3,5,7,9],
        [2,3,5,6,8,9],
        [0,1,4,6,7,9],
        [0,1,2,4,5,7,9],
        [1,3,4,5,6,7,9],
        [0,1,2,3,5,6,8],
        [1,2,3,5,6,8,9],
        [0,2,3,4,6,7,8],
        [1,2,3,4,5,7,8,9],
        [0,1,2,3,4,6,7,9],
        [0,1,2,4,5,6,8,9],
        [0,2,3,4,5,6,8,9],
        [0,1,3,4,5,6,7,8],
        [0,1,2,3,4,5,6,7,8],
        [0,1,2,3,4,5,7,8,9],
        [0,1,2,3,4,6,7,8,9],
        [0,1,2,3,4,5,6,7,9],
        [0,1,2,3,4,5,6,8,9]
    ]
    '''
    
    '''
    for i in range(len(hide_shapes)):
        hide_shape_one = hide_shapes[i]
        hide_color_one = hide_colors[i]
        if i+1 < len(hide_shapes):
            hide_shape_two = hide_shapes[i+1]
            hide_color_two = hide_colors[i+1]
        else:
            hide_shape_two = None
            hide_color_two = None
        if i+2 < len(hide_shapes):
            hide_shape_three = hide_shapes[i+2]
            hide_color_three = hide_colors[i+2]
        else:
            hide_shape_three = None
            hide_color_three = None
    '''   

    uns_percentages=[1.0]
    for uns_percentage in uns_percentages:
        parameter_sets = [
            # ^ first node
            {
                'model_parameter_name': 'shieldedmnist',
                'dataset_parameter_name': 'shortmnist',
                'uns_parameter_percentage': uns_percentage,
                'GPU_ID': '3',
                'my_task': 'addition',
                'NA': False,  # do not use augmentations for mnist
                'hide_shapes_parameter': None,
                'hide_colors_parameter': None,
            }
        ]

        run_parallel_executions(parameter_sets)