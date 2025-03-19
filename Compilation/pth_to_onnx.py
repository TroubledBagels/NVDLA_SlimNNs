import sys

import torch
import onnx

from AlexSNN import AlexSNN

from onnx import version_converter


def open_model(model_name: str) -> AlexSNN:
    return torch.load(f"./snn_models/{model_name}.pth")


def to_onnx(model: AlexSNN, model_name: str) -> bool:
    print("Doing initial export...")
    dummy_input = torch.randn(1, *model.input_shape).cuda()
    torch.onnx.export(
        model,
        dummy_input,
        f"./onnx_models/{model_name}.onnx",
        input_names=["input"],
        output_names=["output"],
        export_params=True,
        verbose=True,
        opset_version=8
    )

    print("Downgrading...")
    to_downgrade = onnx.load(f"./onnx_models/{model_name}.onnx")

    converted_model = version_converter.convert_version(to_downgrade, 8)

    onnx.save(converted_model, f"./onnx_models/{model_name}.onnx")

    print("Checking...")
    check_model = onnx.load(f"./onnx_models/{model_name}.onnx")
    try:
        onnx.checker.check_model(check_model)
    except Exception as e:
        print(e)
        return False
    print("Initialisers in the ONNX mode:")
    for init in check_model.graph.initializer:
        print(f"    Name: {init.name}, Shape: {init.dims}")

    print("\nNodes in the ONNX model:")
    for node in check_model.graph.node:
        print(f"    Name: {node.name}, Node: {node.op_type}, Inputs: {node.input}, Outputs: {node.output}")

    print("\nParams and Gradient:")
    for name, param in model.named_parameters():
        print(f"    Parameter: {name}, Shape: {param.shape}, Requires_Grad: {param.requires_grad}")

    print()
    print(onnx.helper.printable_graph(check_model.graph))

    return True

if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 2:
        print("Usage: python3 pth_to_onnx.py <model_name>")
    model = open_model(sys.argv[1])
    # if isinstance(model, SNN):
    #     SNN.input_shape = (3, 32, 32)
    model.eval()
    for name, param in model.named_parameters():
        print(name, param.shape)
    if to_onnx(model, sys.argv[1]):
        print("Success!")
    pass
