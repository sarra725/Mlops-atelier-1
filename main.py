import argparse
from model_pipeline import (
    load_data,
    clean_data,
    encode_data,
    split_data,
    normalize_data,
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model,
    predict,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true", help="Prepare the data")
    parser.add_argument("--train", action="store_true", help="Train and save the model")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the model")
    parser.add_argument("--predict", action="store_true", help="Predict on test data")
    parser.add_argument("--load", action="store_true", help="Load the model")
    args = parser.parse_args()

    if args.prepare:
        prepare_data()
        print("Data prepared.")

    if args.train:
        x_train, x_test, y_train, y_test = prepare_data()
        model = train_model(x_train, y_train)
        save_model(model)
        print("Model trained and saved.")

    if args.evaluate:
        x_train, x_test, y_train, y_test = prepare_data()
        model = load_model()
        evaluate_model(model, x_test, y_test)

    if args.predict:
        # predict() normalise lui-même : on lui passe les données brutes
        data = encode_data(clean_data(load_data()))
        x_train, x_test, y_train, y_test = split_data(data)
        predictions = predict(x_test)
        print(predictions[:10])

    if args.load:
        load_model()


if __name__ == "__main__":
    main()
