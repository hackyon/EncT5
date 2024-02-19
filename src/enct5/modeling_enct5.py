# coding=utf-8
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" EncT5 model (based on HuggingFace T5 Model) """

from typing import Optional, List, Tuple, Union

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.models.t5.modeling_t5 import T5Config, T5PreTrainedModel, T5Model
from transformers.modeling_outputs import Seq2SeqSequenceClassifierOutput

from .configuration_enct5 import EncT5Config


class EncT5ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config: EncT5Config):
        super().__init__()
        self.dropout = nn.Dropout(p=config.classifier_dropout)
        self.out_proj = nn.Linear(config.d_model, config.num_labels)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states


class EncT5MultiLabelClassificationHead(nn.Module):
    """Head for multi-label sentence-level classification tasks."""

    def __init__(self, config: EncT5Config):
        super().__init__()
        self.weights = nn.Parameter(torch.Tensor(config.num_labels, config.d_model))
        self.biases = nn.Parameter(torch.Tensor(config.num_labels))
        self.dropout = nn.Dropout(p=config.classifier_dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # The input hidden_states shape should be (batch_size, num_labels, d_model)
        hidden_states = self.dropout(hidden_states)
        # The following element-wise multiplication simulates multiple per-label classification heads (one head per
        # label). The element-wise multiplication of the weights, followed by a summation and addition of biases, is
        # equivalent to a linear projection from d_model down to 1 for each label (but with vectorization).
        hidden_states = torch.sum(hidden_states * self.weights, dim=-1) + self.biases  # (batch_size, num_labels)
        return hidden_states


class EncT5PreTrainedModel(T5PreTrainedModel):
    def _init_weights(self, module):
        """Initialize the weights"""
        factor = self.config.initializer_factor  # Used for testing weights initialization
        if isinstance(module, EncT5ClassificationHead):
            module.out_proj.weight.data.normal_(mean=0.0, std=factor * (self.config.d_model ** -0.5))
            if hasattr(module.out_proj, "bias") and module.out_proj.bias is not None:
                module.out_proj.bias.data.zero_()
        elif isinstance(module, EncT5MultiLabelClassificationHead):
            module.weights.data.normal_(mean=0.0, std=factor * ((self.config.d_model) ** -0.5))
            module.biases.data.zero_()


class EncT5ForSequenceClassification(EncT5PreTrainedModel):
    r"""
    The EncT5 model was proposed in [EncT5: A Framework for Fine-tuning T5 as Non-autoregressive
    Models](https://arxiv.org/abs/2110.08426) by Frederick Liu, Terry Huang, Shihang Lyu, Siamak Shakeri, Hongkun Yu,
    Jing Li.

    EncT5 is a variant of T5 that uses mainly the encoder for non-autoregressive tasks. There are several special
    features to EncT5: 1) there are less decoder layers (defaulting to 1 decoder layer), 2) there is a separate decoder
    word embedding, with the decoder input ids being predefined constants, and 3) there is a classification head on top
    of the output. Research has shown that this model can be more efficient and usable over T5 and BERT for
    non-autoregressive tasks such as classification and regression.
    """
    config_class = EncT5Config
    _keys_to_ignore_on_load_unexpected = ["decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight"]

    def __init__(self, config: EncT5Config):
        super().__init__(config)

        # Initialize the base T5 model.
        self.transformer = T5Model(T5Config.from_dict(config.to_dict()))

        # Initiate decoder embedding from scratch and define the corresponding latent vector vocabulary size.
        self.decoder_embeddings = nn.Embedding(config.decoder_vocab_size, config.d_model)
        self.transformer.get_decoder().set_input_embeddings(self.decoder_embeddings)

        # Initiate decoder projection head from scratch.
        if config.problem_type == "multi_label_classification":
            self.classification_head = EncT5MultiLabelClassificationHead(config)
        else:
            self.classification_head = EncT5ClassificationHead(config)

        # Initialize weights and apply final processing
        self.post_init()

        self.model_parallel = False

    def load_weights_from_pretrained_t5(self, model_path: str):
        pretrained_t5_model = T5Model.from_pretrained(model_path)

        # Override the decoder embedding weights to make them the correct shape.
        pretrained_state_dict = pretrained_t5_model.state_dict()
        pretrained_state_dict["decoder.embed_tokens.weight"] = self.decoder_embeddings.state_dict()["weight"]

        self.transformer.load_state_dict(pretrained_state_dict, strict=False)

    def prepare_for_fine_tuning(self):
        r"""
        Prepares the model for fine-tuning by re-initializing the necessary weights for fine-tuning. This step should be
        performed after loading the pre-trained T5 model but before fine-tuning.
        """
        self.decoder_embeddings.weight.data.normal_(mean=0.0, std=self.config.initializer_factor)
        self._init_weights(self.classification_head)

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            decoder_input_ids: Optional[torch.LongTensor] = None,
            decoder_attention_mask: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            decoder_head_mask: Optional[torch.Tensor] = None,
            cross_attn_head_mask: Optional[torch.Tensor] = None,
            encoder_outputs: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Seq2SeqSequenceClassifierOutput]:
        r"""
        Arguments:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. T5 is a model with relative position embeddings so
                you should be able to pad the inputs on both the right and the left.

                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for detail.

                [What are input IDs?](../glossary#input-ids)

                To know more on how to prepare `input_ids` for pretraining take a look a [T5 Training](./t5#training).
            attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            decoder_input_ids (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
                Indices of decoder input sequence tokens in the vocabulary.

                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are decoder input IDs?](../glossary#decoder-input-ids)

                T5 uses the `pad_token_id` as the starting token for `decoder_input_ids` generation. If
                `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
                `past_key_values`).

                To know more on how to prepare `decoder_input_ids` for pretraining take a look at [T5
                Training](./t5#training).
            decoder_attention_mask (`torch.BoolTensor` of shape `(batch_size, target_sequence_length)`, *optional*):
                Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will
                also be used by default.
            head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
                Mask to nullify selected heads of the self-attention modules in the encoder. Mask values selected in
                `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            decoder_head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
                Mask to nullify selected heads of the self-attention modules in the decoder. Mask values selected in
                `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            cross_attn_head_mask (`torch.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
                    Mask to nullify selected heads of the cross-attention modules in the decoder. Mask values selected
                    in `[0, 1]`:

                    - 1 indicates the head is **not masked**,
                    - 0 indicates the head is **masked**.

            encoder_outputs (`tuple(tuple(torch.FloatTensor)`, *optional*):
                Tuple consists of (`last_hidden_state`, `optional`: *hidden_states*, `optional`: *attentions*)
                `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)` is a sequence of hidden states
                at the output of the last layer of the encoder. Used in the cross-attention of the decoder.
            past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4
                tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
                Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up
                decoding.

                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those
                that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of
                all `decoder_input_ids` of shape `(batch_size, sequence_length)`.
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            decoder_inputs_embeds (`torch.FloatTensor` of shape `(batch_size, target_sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `decoder_input_ids` you can choose to directly pass an embedded
                representation. If `past_key_values` is used, optionally only the last `decoder_inputs_embeds` have to
                be input (see `past_key_values`). This is useful if you want more control over how to convert
                `decoder_input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

                If `decoder_input_ids` and `decoder_inputs_embeds` are both unset, `decoder_inputs_embeds` takes the
                value of `inputs_embeds`.
            labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
                config.num_labels - 1]`. If `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
                tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
                more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        Returns:
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if labels is not None:
            use_cache = False

        if input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds.")
        batch_size = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if decoder_input_ids is None and decoder_inputs_embeds is None:
            if self.config.problem_type == "multi_label_classification":
                decoder_input_ids = torch.arange(end=self.config.num_labels, device=device, dtype=torch.long)
                decoder_input_ids = decoder_input_ids.repeat(batch_size, 1)  # Shape: (batch_size, num_labels)
                # Provide a 3-dimensional attention mask by default to suppress the default causal mask.
                if decoder_attention_mask is None:
                    decoder_attention_mask = torch.ones(
                        (batch_size, self.config.num_labels, self.config.num_labels), device=device, dtype=torch.long
                    )
            else:
                decoder_input_ids = torch.zeros(batch_size, 1, device=device, dtype=torch.long)

        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]  # Shape: (batch_size, 1 or num_labels, d_model)

        logits = self.classification_head(sequence_output)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.config.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.config.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    # The classification head for multi-label classification is different, and so we need the
                    # problem_type to be set during initialization to select the proper classification head.
                    raise ValueError(
                        "For multi-label classification, the config.problem_type must be set to "
                        "'multi_label_classification' when initializing the model.")

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.config.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
            else:
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )
