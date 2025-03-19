from snn_dep import graphing_deps, snn_data
import os
import torch
import sys
from AlexSNN import AlexSNN

def main(model_name: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    wml = []

    if "2_part" in model_name:
        wml = [0.5, 1.0]
    elif "4_part" in model_name:
        wml = [0.25, 0.5, 0.75, 1.0]
    else:
        wml = [0.5, 1.0]

    print(wml)

    model = torch.load(f"./snn_models/{model_name}.pth")
    _, test_dl = snn_data.load_data("CIFAR10", 1, 3)
    threshold_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99]
    correct, confidence, times, widths = graphing_deps.run_thresholding_tests(model, test_dl, device, 3, threshold_list)
    plots = graphing_deps.create_graphs(correct, confidence, times, widths, threshold_list, wml)
    # Show plots
    # plt.show()
    graphing_deps.export_graphs(model_name, plots)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 generate_graphs.py <model_name>")
        sys.exit(1)
    main(sys.argv[1])