import os
import pandas as pd

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

airplane_means = []
automobile_means = []
bird_means = []
cat_means = []
deer_means = []
dog_means = []
frog_means = []
horse_means = []
ship_means = []
truck_means = []

for i in range(len(names)):
    print(f"Mean error for {names[i]}: {mean_errors_per_image[i]}")
    if "airplane" in names[i]:
        airplane_means.append(round(mean_errors_per_image[i], 5))
    elif "automobile" in names[i]:
        automobile_means.append(round(mean_errors_per_image[i], 5))
    elif "bird" in names[i]:
        bird_means.append(round(mean_errors_per_image[i], 5))
    elif "cat" in names[i]:
        cat_means.append(round(mean_errors_per_image[i], 5))
    elif "deer" in names[i]:
        deer_means.append(round(mean_errors_per_image[i], 5))
    elif "dog" in names[i]:
        dog_means.append(round(mean_errors_per_image[i], 5))
    elif "frog" in names[i]:
        frog_means.append(round(mean_errors_per_image[i], 5))
    elif "horse" in names[i]:
        horse_means.append(round(mean_errors_per_image[i], 5))
    elif "ship" in names[i]:
        ship_means.append(round(mean_errors_per_image[i], 5))
    elif "truck" in names[i]:
        truck_means.append(round(mean_errors_per_image[i], 5))

print(f"Mean error for airplane: {sum(airplane_means) / 10}")
print(f"Mean error for automobile: {sum(automobile_means) / 10}")
print(f"Mean error for bird: {sum(bird_means) / 10}")
print(f"Mean error for cat: {sum(cat_means) / 10}")
print(f"Mean error for deer: {sum(deer_means) / 10}")
print(f"Mean error for dog: {sum(dog_means) / 10}")
print(f"Mean error for frog: {sum(frog_means) / 10}")
print(f"Mean error for horse: {sum(horse_means) / 10}")
print(f"Mean error for ship: {sum(ship_means) / 10}")
print(f"Mean error for truck: {sum(truck_means) / 10}")
# print(f"Mean error per image: {mean_errors_per_image}")
print("Average error: " + str(error_sum / num_points))

column_names = ["Airplane", "Automobile", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"]
output_df = pd.DataFrame(columns=column_names)

for i in range(len(airplane_means)):
    row = [airplane_means[i], automobile_means[i], bird_means[i], cat_means[i], deer_means[i], dog_means[i], frog_means[i], horse_means[i], ship_means[i], truck_means[i]]
    output_df.loc[i] = row

row = [(sum(airplane_means) / 10), (sum(automobile_means) / 10), (sum(bird_means) / 10), (sum(cat_means) / 10), (sum(deer_means) / 10), (sum(dog_means) / 10), (sum(frog_means) / 10), (sum(horse_means) / 10), (sum(ship_means) / 10), (sum(truck_means) / 10)]
output_df.loc[len(output_df)] = row

output_df.to_csv("mean_errors.csv")
