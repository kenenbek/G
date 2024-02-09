from mydata import generate_train_test_indices, MyDataset

if __name__ == '__main__':
    path = "scand"
    full_dataset = MyDataset(root=f"full_data/{path}/", dataset=f"{path}")
    full_data = full_dataset[0]
    print(full_data.validate(raise_on_error=False))
    generate_train_test_indices(y=full_data.y, path=path)

