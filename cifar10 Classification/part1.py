import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import os


def dataset_to_numpy(dataset):
    X = []
    y = []

    for img, label in dataset:
        # flatten each image into a 3072-long vector
        X.append(img.numpy().reshape(-1))
        y.append(label)

    return np.array(X), np.array(y)


def load_images_from_folder(folder_path, transform):
    image_extensions = (".jpg", ".jpeg", ".png")
    image_files = [
        filename
        for filename in os.listdir(folder_path)
        if filename.lower().endswith(image_extensions)
    ]

    X = []
    filenames = []

    for filename in image_files:
        full_path = os.path.join(folder_path, filename)
        with Image.open(full_path) as img:
            img = img.convert("RGB")
            # resize external images to CIFAR-10 size, then flatten
            X.append(transform(img).numpy().reshape(-1))
            filenames.append(filename)

    if not X:
        return np.empty((0, 3072)), []

    return np.array(X), filenames


def evaluate_cat_folder(knn, folder_path, X_train, y_train_raw, class_names, top_k=2):
    external_image_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])

    X_cat_folder, cat_filenames = load_images_from_folder(folder_path, external_image_transform)

    if len(X_cat_folder) == 0:
        print("No images found in cat folder.")
        return 0.0

    y_cat_pred = knn.predict(X_cat_folder)
    cat_folder_accuracy = accuracy_score(np.ones_like(y_cat_pred), y_cat_pred)

    print("Accuracy on cat images:", cat_folder_accuracy)

    # Get nearest training sample for each external cat image.
    distances, neighbor_indices = knn.kneighbors(X_cat_folder, n_neighbors=1, return_distance=True)
    distances = distances.ravel()
    neighbor_indices = neighbor_indices.ravel()

    # Show the strongest matches (smallest distance).
    best_count = min(top_k, len(X_cat_folder))
    best_order = np.argsort(distances)[:best_count]

    print(f"\nTop {best_count} nearest matches for external cat images:")
    for rank, ext_idx in enumerate(best_order, start=1):
        train_idx = neighbor_indices[ext_idx]
        train_class = class_names[y_train_raw[train_idx]]
        print(
            f"  {rank}. {cat_filenames[ext_idx]} -> train idx {train_idx} "
            f"({train_class}), distance={distances[ext_idx]:.4f}"
        )

    fig, axes = plt.subplots(best_count, 2, figsize=(8, 4 * best_count))
    if best_count == 1:
        axes = np.array([axes])

    for row, ext_idx in enumerate(best_order):
        train_idx = neighbor_indices[ext_idx]
        ext_img = np.transpose(X_cat_folder[ext_idx].reshape(3, 32, 32), (1, 2, 0))
        match_img = np.transpose(X_train[train_idx].reshape(3, 32, 32), (1, 2, 0))

        axes[row, 0].imshow(np.clip(ext_img, 0, 1))
        axes[row, 0].set_title(f"External: {cat_filenames[ext_idx]}")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(np.clip(match_img, 0, 1))
        axes[row, 1].set_title(
            f"Nearest train match\nclass={class_names[y_train_raw[train_idx]]}, d={distances[ext_idx]:.4f}"
        )
        axes[row, 1].axis("off")

    plt.tight_layout()
    plt.savefig("cat_best_matches.png", dpi=150, bbox_inches="tight")
    print("Saved side-by-side matches to cat_best_matches.png")
    plt.show()


if __name__ == "__main__":
    transform = transforms.ToTensor()

    train_data = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    test_data = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    class_names = train_data.classes

    X_train, y_train_raw = dataset_to_numpy(train_data)
    X_test, y_test_raw = dataset_to_numpy(test_data)

    # binary labels: cat = 1, not cat = 0
    cat_class = 3
    y_train = (y_train_raw == cat_class).astype(int)
    y_test = (y_test_raw == cat_class).astype(int)

    subset_size = 50000  # use all training samples
    X_train = X_train[:subset_size]
    y_train = y_train[:subset_size]

    # k-NN classifier using Euclidean distance
    knn = KNeighborsClassifier(
        n_neighbors=1,
        metric="euclidean"
    )

    # train classifier
    knn.fit(X_train, y_train)

    # predict on CIFAR-10 test set
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # accuracy on cat test images only
    cat_idx = np.where(y_test == 1)[0]
    cat_accuracy = accuracy_score(y_test[cat_idx], y_pred[cat_idx])

    print("Part 1 - KNN cat classification")
    print("Training samples used:", len(X_train))
    print("Overall test accuracy:", accuracy)
    print("Accuracy on cat test images:", cat_accuracy)

    evaluate_cat_folder(
        knn,
        folder_path=os.path.join(os.path.dirname(__file__), "cat"),
        X_train=X_train,
        y_train_raw=y_train_raw,
        class_names=class_names,
        top_k=2,
    )