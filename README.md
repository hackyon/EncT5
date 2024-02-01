# EncT5

Implementation of `EncT5` based
on [Fine-tuning T5 Encoder for Non-autoregressive Tasks](https://arxiv.org/abs/2110.08426).

EncT5 is a variant of T5 that utilizes mainly the encoder for non-autoregressive (ie. classification and regression)
tasks. It uses the same base weights at T5, but requires fine-tuning before use. There are several special features to
EncT5:

1. There are less decoder layers (a single decoder layer by default), and so saves on parameters/resources.
2. There is a separate decoder word embedding, with the decoder input ids being predefined constants. During
   fine-tuning, these constants are trained to effectively "prompt" the encoder to perform the necessary
   classification/regression tasks.
3. There is a classification head on top of the decoder output.

Research has shown that this model can be more efficient and usable over T5 and BERT for non-autoregressive
tasks such as classification and regression.

## Quickstart

Here is an example of fine-tuning and validating EncT5 over SST2 (positive/negative sentiment analysis over
sentences) in the [GLUE](https://huggingface.co/datasets/glue) dataset.

First, we load the train dataset and use it to fine-tune the EncT5 model:

    # Load training SST2 dataset from GLUE
    train_dataset = load_dataset("glue", "sst2", split="train")
    metric = evaluate.load("glue", "sst2")

    # Perform tokenization of the input.
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    def tokenize_function(sample):
        return tokenizer(sample["sentence"], truncation=True)
    tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Load the EncT5 model from t5-base and prepare it for fine tuning
    config = T5Config.from_pretrained("t5-base", problem_type="single_label_classification", num_labels=2,
                                      num_decoder_layers=1, decoder_vocab_size=1)
    model = EncT5.from_pretrained("t5-base", config=config)
    model.prepare_for_fine_tuning()

    # Define the compute metrics function for training
    def compute_metrics(eval_preds):
        output, labels = eval_preds
        logits = output[0]
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    # Fine-tune the model
    training_args = TrainingArguments("glue-sst2-trainer", evaluation_strategy="epoch")
    trainer = Trainer(
        model,
        training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_validation_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model("./enct5-glue-sst2/")

Then, we loaded the fine-tuned model and evaluate it with the validation dataset:

    # Load validation SST2 dataset from GLUE
    metric = evaluate.load("glue", "sst2")
    validation_dataset = load_dataset("glue", "sst2", split="validation")

    # Perform tokenization of the input.
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    tokenized_validation_dataset = tokenizer(validation_dataset["sentence"], return_tensors="pt",
                                             truncation=True, padding=True)

    # Load the fine-tuned EncT5 model.
    model = EncT5.from_pretrained("./enct5-glue-sst2/")

    # Predict
    output = model(tokenized_validation_dataset["input_ids"])
    logits = output[0]

    # Select the label with the largest logits value (skipping softmax) 
    predictions = np.argmax(logits.detach(), axis=-1) 

    # Compute and output metric
    metric = evaluate.load("glue", "sst2")
    print(metric.compute(predictions=predictions, references=validation_dataset["label"]))

## Installation

    pip install -r requirements.txt