# EncT5

Implementation of `EncT5` based
on [Fine-tuning T5 Encoder for Non-autoregressive Tasks](https://arxiv.org/abs/2110.08426).

EncT5 is a variant of T5 that utilizes mainly the encoder for non-autoregressive (ie. classification and regression)
tasks. It uses the same base weights at T5, but **must be fine-tuning before use**. There are several special features
to EncT5:

1. There are less decoder layers (a single decoder layer by default), and so has fewer parameters/resources than the
   standard T5.
2. There is a separate decoder word embedding, with the decoder input ids being predefined constants. During
   fine-tuning, the decoder embedding learns to use these constants as "prompts" to the encoder for the corresponding
   classification/regression tasks.
3. There is a classification head on top of the decoder output.

Research has shown that this model can be more efficient and usable over T5 and BERT for non-autoregressive
tasks such as classification and regression.

This model (the t5-base variant) can be found on the [HuggingFace model hub](https://huggingface.co/hackyon/enct5-base).

## Quickstart

### Running from HuggingFace Model Hub (Remote Code)

Here is an example of fine-tuning and validating EncT5 over SST2 (positive/negative sentiment analysis over
sentences) in the [GLUE](https://huggingface.co/datasets/glue) dataset.

First, we load the train dataset and use it to fine-tune the EncT5 model:

    # Load training SST2 dataset from GLUE.
    train_dataset = load_dataset("glue", "sst2", split="train")
    metric = evaluate.load("glue", "sst2")

    # Perform tokenization of the input.
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    def tokenize_function(sample):
        return tokenizer(sample["sentence"], truncation=True)
    tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Load the EncT5ForSequenceClassification model from HuggingFace Model Hub.
    # It is recommended that you pin the model to a certain revision. See
    # https://huggingface.co/docs/transformers/custom_models#using-a-model-with-custom-code
    model = AutoModelForSequenceClassification.from_pretrained("hackyon/enct5-base", trust_remote_code=True)

    # Define the compute metrics function for training.
    def compute_metrics(eval_preds):
        output, labels = eval_preds
        logits = output[0]
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    # Fine-tune the model.
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
    model = AutoModelForSequenceClassification.from_pretrained("./enct5-glue-sst2/")

    # Predict
    output = model(tokenized_validation_dataset["input_ids"])
    logits = output[0]

    # Select the label with the largest logits value (skipping softmax).
    predictions = np.argmax(logits.detach(), axis=-1) 

    # Compute and output metric.
    metric = evaluate.load("glue", "sst2")
    print(metric.compute(predictions=predictions, references=validation_dataset["label"]))



### Running from Source on Github

The code for training is the same, but you will need to load the model using the following code instead:

    # Copy the base values from the original T5 config.
    t5_config = T5Config.from_pretrained("t5-base")
    config = EncT5Config.from_dict(t5_config.to_dict())

    # Configure the values specifically for EncT5. The following are the recommended config values.
    config.num_decoder_layers = 1

    config.problem_type = "single_label_classification"
    config.num_labels = 2
    config.decoder_vocab_size = 1

    # Create the model, load the weights from the original T5, and prepare it for fine-tuning.
    model = EncT5ForSequenceClassification(config)
    model.load_weights_from_pretrained_t5("t5-base")
    model.prepare_for_fine_tuning()

    # Continue the fine-tuning...

## Installation

    pip install -r requirements.txt