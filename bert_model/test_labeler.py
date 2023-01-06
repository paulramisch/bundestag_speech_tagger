from simpletransformers.classification import ClassificationModel
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def set_bool(row):
    if row['classifier_output_int'] == 1:
        return True
    else:
        return False

# oad models with and without attention
if __name__ == '__main__':
    # model = ClassificationModel("bert", "outputs/checkpoint-2000", use_cuda=torch.cuda.is_available())
    model = ClassificationModel("bert", "outputs/checkpoint-8312-epoch-4", use_cuda=False)
    # model = ClassificationModel("bert", "outputs/checkpoint-10390-epoch-5", use_cuda=False)

    # Example strings
    # strings = ["Die Leben ist schön", ") Dr. Wahl (CPU)", "Dr. Wahl (CDU)", "Dr. Töpfer, Bundesminister für Umwelt, Naturschutz und Reaktorsicherheit"]
    # print(model.predict(strings))

    # Open annotated data
    annotated_data = pd.read_csv("../data/comparison_data.csv")

    # Counter
    i = 0
    chunk_size = 500
    len_annotated_data = len(annotated_data)

    # Make prediction
    prediction_df_columns = ['classifier_output_int', 'int']
    prediction_df_list = []


    for chunk in chunks(annotated_data["string"].values.tolist(), chunk_size):
        prediction_chunk = model.predict(chunk)

        prediction_chunk_df = pd.DataFrame(list(zip(prediction_chunk[0], prediction_chunk[1])), columns=prediction_df_columns)
        prediction_df_list.append(prediction_chunk_df)

        i += chunk_size
        print(i, "/", len_annotated_data)

    prediction = pd.concat(prediction_df_list, ignore_index=True)
    prediction["classifier_output"] = prediction.apply(lambda row: set_bool(row), axis=1)


    # Combine the two dataframes
    annotated_data = pd.concat([annotated_data, prediction], axis=1, join="inner")

    # Find mistakes
    mistakes = annotated_data[annotated_data["classifier_output"] != annotated_data["is_speech"]]
    print(len(mistakes), " - ", round(len(mistakes) / len(annotated_data), 4))
    print(mistakes)

    # Save mistakes dataframe
    mistakes.to_csv("test_mistakes.csv")
