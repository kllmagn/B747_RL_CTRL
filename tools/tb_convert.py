from general import write_dataframe

def convert_tb_data(root_dir, sort_by=None, ignore_runs=[]):
    """Convert local TensorBoard data into Pandas DataFrame.
    
    Function takes the root directory path and recursively parses
    all events data.    
    If the `sort_by` value is provided then it will use that column
    to sort values; typically `wall_time` or `step`.
    
    *Note* that the whole data is converted into a DataFrame.
    Depending on the data size this might take a while. If it takes
    too long then narrow it to some sub-directories.
    
    Paramters:
        root_dir: (str) path to root dir with tensorboard data.
        sort_by: (optional str) column name to sort by.
    
    Returns:
        pandas.DataFrame with [wall_time, name, step, value] columns.
    
    """
    import os
    import pandas as pd
    from tensorflow.python.summary.summary_iterator import summary_iterator

    def convert_tfevent(filepath):
        return pd.DataFrame([
            parse_tfevent(e) for e in summary_iterator(filepath) if len(e.summary.value)
        ])

    def parse_tfevent(tfevent):
        return dict(
            wall_time=tfevent.wall_time,
            name=tfevent.summary.value[0].tag,
            step=tfevent.step,
            value=float(tfevent.summary.value[0].simple_value),
        )
    
    out = []
    for (root, _, filenames) in os.walk(root_dir):
        for filename in filenames:
            dir_name = os.path.basename(root)
            if "events.out.tfevents" not in filename:
                continue
            if any([ignore in dir_name for ignore in ignore_runs]):
                continue
            file_full_path = os.path.join(root, filename)
            df = convert_tfevent(file_full_path)
            df = pd.pivot_table(df, values = 'value', index = ['step'], columns = ['name'])
            df.columns = [f"{column}__{dir_name}" for column in df.columns]
            out.append(df)


    # Concatenate (and sort) all partial individual dataframes
    all_df = pd.concat(out, axis=1)
    if sort_by is not None:
        all_df = all_df.sort_values(sort_by)
        
    return all_df

if __name__ == "__main__":
    dir_path = "./.logs/tb_log"
    output_path = "tensorboard.xlsx"

    df = convert_tb_data(dir_path, ignore_runs=["AERO_DISTURBANCE"])
    write_dataframe(df, output_path)
    write_dataframe(df, 'tensorboard_big.xlsx', bigMode=True)