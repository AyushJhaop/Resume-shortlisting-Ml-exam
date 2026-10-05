import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/resume_shortlisting.csv")

print("Shape:", df.shape)
print("\nData types:\n", df.dtypes)
print("\nMissing values:\n", df.isnull().sum())
print("\nTarget distribution:\n", df["shortlisted"].value_counts())
print("\nSummary:\n", df.describe(include="all").T)

df["shortlisted"].value_counts().plot(kind="bar")
plt.title("Shortlisting Class Distribution")
plt.xlabel("Shortlisted (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

df.groupby("education_level")["shortlisted"].mean().sort_values().plot(kind="bar")
plt.title("Shortlisting Rate by Education")
plt.ylabel("Shortlisting Rate")
plt.tight_layout()
plt.show()

df.groupby("previous_industry")["shortlisted"].mean().sort_values().plot(kind="bar")
plt.title("Shortlisting Rate by Previous Industry")
plt.ylabel("Shortlisting Rate")
plt.tight_layout()
plt.show()
