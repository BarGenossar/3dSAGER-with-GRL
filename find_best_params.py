import os
import pickle


def find_best_result(results_dir="results"):
    best_f1 = -1.0
    best_file = None
    best_params = None

    for file in os.listdir(results_dir):
        if not file.endswith(".pkl"):
            continue

        path = os.path.join(results_dir, file)
        with open(path, "rb") as f:
            result = pickle.load(f)

        f1_mean = result["summary"]["f1"]["mean"]
        if f1_mean > best_f1:
            best_f1 = f1_mean
            best_file = file
            best_params = result["training_params"]

    print("=======================================")
    print(f"Best file: {best_file}")
    print(f"Best mean F1: {best_f1:.3f}")
    print(f"Recall: {result['summary']['recall']['mean']:.3f}")
    print(f"Precision: {result['summary']['precision']['mean']:.3f}")
    print("Training parameters:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")
    print("=======================================")


if __name__ == "__main__":
    find_best_result("results")
