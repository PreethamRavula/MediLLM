MODEL_PATH_TEXT=medi_llm_state_dict_text.pth
MODEL_PATH_IMAGE=medi_llm_state_dict_image.pth
MODEL_PATH_MULTIMODAL=medi_llm_state_dict_multimodal.pth
SCRIPT=inference.py

# Run inference for multimodal model
infer_multimodal:
	python $(SCRIPT) \
		--mode multimodal \
		--model_path $(MODEL_PATH_MULTIMODAL) \

# Run inference for image-only model
infer_image:
	python $(SCRIPT) \
		--mode image \
		--model_path $(MODEL_PATH_IMAGE) \

# Run inference for text-only model
infer_text:
	python $(SCRIPT) \
		--mode text \
		--model_path $(MODEL_PATH_TEXT) \

# Save only misclassified samples for multimodal model
infer_misclassified:
	python $(SCRIPT) \
		--mode multimodal \
		--model_path $(MODEL_PATH_MULTIMODAL) \
		--save_misclassified_only