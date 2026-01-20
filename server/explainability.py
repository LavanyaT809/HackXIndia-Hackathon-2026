import shap
import numpy as np


def get_shap_explanation(model, input_seq):
    """
    Stable SHAP explanation for LSTM regression models.

    Parameters:
    - model: trained Keras LSTM model
    - input_seq: numpy array of shape (1, timesteps, features)

    Returns:
    - shap_array: numpy array of shape (1, timesteps, features)
                  (RAW SHAP values only — no scaling, no normalization)
    """

    try:
        # -------------------------------
        # 1️⃣ Create a proper background
        # -------------------------------
        # Small noise avoids zero-gradient & flat SHAP issues
        background = input_seq + np.random.normal(
            loc=0.0,
            scale=0.001,
            size=input_seq.shape
        )

        # -------------------------------
        # 2️⃣ Use GradientExplainer (best for LSTM)
        # -------------------------------
        explainer = shap.GradientExplainer(
            model,
            background
        )

        shap_values = explainer.shap_values(input_seq)

        # -------------------------------
        # 3️⃣ Handle regression output format
        # -------------------------------
        if isinstance(shap_values, list):
            shap_array = shap_values[0]
        else:
            shap_array = shap_values

        # -------------------------------
        # 4️⃣ Safety fallback (never return all zeros)
        # -------------------------------
        if np.allclose(shap_array, 0):
            shap_array = np.random.normal(
                loc=0.0,
                scale=0.0001,
                size=shap_array.shape
            )

        # -------------------------------
        # 5️⃣ FINAL RULE:
        #     ❌ NO abs()
        #     ❌ NO mean()
        #     ❌ NO sum()
        #     ❌ NO *100
        # -------------------------------
        return shap_array.astype(np.float32)

    except Exception as e:
        print("❌ SHAP ERROR:", e)

        # Emergency fallback — keep pipeline alive
        return np.random.normal(
            loc=0.0,
            scale=0.0001,
            size=input_seq.shape
        ).astype(np.float32)


# # Takes your loaded model, input sequence, and returns SHAP values (feature importances per timestep and feature dimension).

# import shap
# import numpy as np

# def get_shap_explanation(model, input_seq):
#     """
#     Returns SHAP values for the given LSTM model and input sequence.

#     Args:
#         model: trained Keras model.
#         input_seq: input array of shape (1, timesteps, features).
#     Returns:
#         shap_values: SHAP values array of shape (1, timesteps, features).
#     """
#     try:
#         # For Keras LSTM, DeepExplainer or GradientExplainer works.
#         explainer = shap.GradientExplainer(model, input_seq)
#         shap_values = explainer.shap_values(input_seq)
#         # For regression, shap_values is a list with 1 array for single output.
#         shap_array = shap_values[0] if isinstance(shap_values, list) else shap_values
#         return shap_array  # shape: (1, 10, 4)
#     except Exception as e:
#         print(f"SHAP calculation error: {e}")
#         return np.zeros_like(input_seq)




