import shap
import abc
import pandas as pd
import matplotlib.pyplot as plt

def log_shap_summary(model_wrapper: abc.ABCMeta,
                     X_sample: pd.DataFrame,
                     feature_names: list,
                     shap_summary_path: str):


    model = model_wrapper.get_model()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    shap_df = pd.DataFrame(shap_values, columns=feature_names)

    shap.summary_plot(shap_values, X_sample,  feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(shap_summary_path)
    plt.close()

    return shap_summary_path, shap_df