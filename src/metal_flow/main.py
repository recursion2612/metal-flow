import qiskit_metal
from metal_flow.design import FOUR_QUBIT_DESIGN_DICT
from metal_flow.functions import create_design
def main():
    print("Hello from metal-flow!")
    print(qiskit_metal.about())
    design = create_design(FOUR_QUBIT_DESIGN_DICT)
    qiskit_metal.view(design).savefig("main_py_test_design.png")


if __name__ == "__main__":

    main()
