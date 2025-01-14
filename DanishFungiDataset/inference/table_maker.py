from typing import Set

import pandas as pd
import wandb


def get_results_df(tags: Set[str]) -> pd.DataFrame:
    """
    Fetches results from Weights & Biases runs filtered by specified tags.

    Parameters
    ----------
    tags : set of str
        Set of tags to filter the runs.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the final metrics of filtered runs.
    """
    api = wandb.Api()
    entity, project = "zcu_cv", "DanishFungi2024"

    filtered_tags = {"tags": {"$in": ["Debug", "Production"]}}

    runs = api.runs(f"{entity}/{project}", filters=filtered_tags)
    results_list = []

    for run in runs:
        print(f"Processing run: {run.name}, state: {run.state}, tags: {run.tags}")  # Debugging statement

        if run.state.lower() != "finished":
            print(f"Skipping run {run.name} because it is not finished.")  # Debugging statement
            continue

        run_tags = run.tags
        if not tags.issubset(run_tags):
            print(f"Skipping run {run.name} because tags do not match.")  # Debugging statement
            continue

        history = run.history()
        metrics = history[["Val. Accuracy", "Val. Recall@3", "Val. F1"]]
        final_metrics = metrics.dropna().iloc[-1].to_dict()
        final_metrics = {k: float(v) for k, v in final_metrics.items()}  # Convert to native Python float
        final_metrics["run_name"] = run.name
        results_list.append(final_metrics)

    # Debugging statement to inspect results_list
    print("Results List:", results_list)

    if not results_list:
        print("No matching runs found.")
        return pd.DataFrame(columns=["Val. Accuracy", "Val. Recall@3", "Val. F1"])

    results_df = pd.DataFrame(results_list)
    results_df.set_index("run_name", inplace=True)
    results_df = results_df[["Val. Accuracy", "Val. Recall@3", "Val. F1"]]

    results_df *= 100
    results_df = results_df.round(decimals=2)

    return results_df


def main():
    """
    Main function to fetch results from Weights & Biases, process, and save them.
    """
    resolution_tag = "299x299"
    dataset_tag = "DF20_FIX"
    output_path = f"../output/{dataset_tag}_{resolution_tag}.txt"
    tags = {resolution_tag, "Production", dataset_tag}

    results_df = get_results_df(tags)
    results_df = results_df.sort_values(["Val. Accuracy", "Val. Recall@3", "Val. F1"])
    results_df.sort_index(inplace=True)

    result_message = (
        f"\n{dataset_tag}-{resolution_tag}: Production\n{results_df.to_markdown()}\n"
    )
    print(result_message)

    with open(output_path, "a") as f:
        f.write(result_message)


if __name__ == "__main__":
    main()
