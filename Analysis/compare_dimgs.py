import os

py_dir = "py_outputs/"
nvdla_dir = "outputs/"

py_files = sorted(os.listdir(py_dir))
nvdla_files = sorted(os.listdir(nvdla_dir))

names = []

all_errors = []

for py_file, nvdla_file in zip(py_files, nvdla_files):
    # load files
    py_file = open(py_dir + py_file, "r")
    nvdla_file = open(nvdla_dir + nvdla_file, "r")

    py_floats = py_file.readlines()[0].split()
    nvdla_floats = nvdla_file.readlines()[0].split()

    file_name = py_file.name.split("/")[-1].split(".")[0]
    names.append(file_name)

    errors = []

    for i in range(len(py_floats)):
        py_floats[i] = float(py_floats[i])
        nvdla_floats[i] = float(nvdla_floats[i])
        errors.append(abs(py_floats[i] - nvdla_floats[i]))

    all_errors.append(errors)
    # close files
    py_file.close()
    nvdla_file.close()

error_sum = 0.0
current_image_sum = 0.0
mean_errors_per_image = []
num_points = 0

for i in range(len(all_errors)):
    for j in range(len(all_errors[i])):
        error_sum += all_errors[i][j]
        current_image_sum += all_errors[i][j]
        num_points += 1
    mean_errors_per_image.append(current_image_sum / len(all_errors[i]))
    current_image_sum = 0.0

for i in range(len(names)):
    print(f"Mean error for {names[i]}: {mean_errors_per_image[i]}")
# print(f"Mean error per image: {mean_errors_per_image}")
print("Average error: " + str(error_sum / num_points))