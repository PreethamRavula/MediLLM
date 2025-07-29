def extract_token_attention(model, tokenizer, input_ids, attention_mask):
    if hasattr(model.text_encoder, 'bert'):
        try:
            outputs = model.text_encoder.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True
            )
            last_attn = outputs.attentions[-1]  # (B, H, S, S), final layer
            weights = last_attn.mean(dim=1)[0, 0, :]  # mean heads, CLS -> token, dim = 1 mean across heads from batch 0, from CLS token, to connection to all other tokens

            weights = weights.detach().cpu().numpy()
            weights = (weights - weights.min()) / (weights.max() - weights.min() + 1e-8)

            tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
            return [(tok, float(round(weights[i], 3))) for i, tok in enumerate(tokens)]

        except Exception as e:
            print("Attention extraction failed:", e)
    return None
