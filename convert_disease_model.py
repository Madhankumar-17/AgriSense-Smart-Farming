
import os
import tensorflow as tf

# Original model
MODEL_PATH = "models/agrisense_disease_model.keras"

# Optimized model output
OUTPUT_PATH = "models/agrisense_disease_model_fp16.tflite"

print("TensorFlow version:", tf.__version__)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

original_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)

print(f"\nOriginal model size: {original_size:.2f} MB")
print("\nLoading original disease model...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("Model loaded successfully!")

print("\nInput shape:", model.input_shape)
print("Output shape:", model.output_shape)

print("\nStarting Float16 conversion...")

converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Enable optimization
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Convert model weights to Float16
converter.target_spec.supported_types = [tf.float16]

# Convert
tflite_model = converter.convert()

# Save converted model
os.makedirs("models", exist_ok=True)

with open(OUTPUT_PATH, "wb") as file:
    file.write(tflite_model)

converted_size = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)

print("\n========== CONVERSION SUCCESS ==========")
print(f"Original model: {original_size:.2f} MB")
print(f"TFLite model:   {converted_size:.2f} MB")
print(f"Space saved:    {original_size - converted_size:.2f} MB")
print(f"Reduction:      {(1 - converted_size / original_size) * 100:.2f}%")
print("\nSaved at:", OUTPUT_PATH)
print("Original model has NOT been modified.")
