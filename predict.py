import argparse


def normalize_model_name(name: str) -> str:
    key = name.strip().lower()
    if key == "transformer":
        return "transformer"
    if key == "bilstm":
        return "bilstm"
    raise ValueError("--model must be one of: Transformer, biLSTM")


def main():
    parser = argparse.ArgumentParser(description="Dispatch prediction script by model type.")
    parser.add_argument("--model", required=True, help="Transformer or biLSTM (case-insensitive)")
    parser.add_argument("--dynamic-csv", default="./Input/example_dynamic_data.csv")
    parser.add_argument("--static-csv", default="./Input/example_static_data.csv")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    model_name = normalize_model_name(args.model)

    if model_name == "transformer":
        from predict_Transformer import main as predictor_main
        predictor_main(args)
    elif model_name == "bilstm":
        from predict_BiLSTM import main as predictor_main
        predictor_main(args)
    else:
        raise ValueError(f"Unsupported model: {model_name}")


if __name__ == "__main__":
    main()