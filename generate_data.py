import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
N = 1200

def generate_dataset(n=N, seed=SEED):
    rng = np.random.default_rng(seed)

    experience = np.clip(rng.normal(4.5, 2.8, n), 0, 15)
    skills = np.clip(rng.poisson(6, n), 0, 15)
    certifications = np.clip(rng.poisson(2, n), 0, 8)

    education = rng.choice(
        ["Bachelor's", "Master's", "PhD"],
        n,
        p=[0.58, 0.34, 0.08]
    )

    industry = rng.choice(
        ["IT", "Finance", "Healthcare", "Retail", "Manufacturing"],
        n,
        p=[0.35, 0.18, 0.15, 0.15, 0.17]
    )

    # Synthetic-data generation rule:
    # experience, skills, certifications and education contribute positively,
    # while industry has a small contextual effect. Noise prevents a perfect rule.
    edu_score = pd.Series(education).map({
        "Bachelor's": 0.0, "Master's": 1.0, "PhD": 1.5
    }).to_numpy()

    industry_score = pd.Series(industry).map({
        "IT": 0.50,
        "Finance": 0.20,
        "Healthcare": 0.10,
        "Retail": -0.20,
        "Manufacturing": -0.10
    }).to_numpy()

    latent_score = (
        -3.0
        + 0.28 * experience
        + 0.16 * skills
        + 0.28 * certifications
        + 0.75 * edu_score
        + industry_score
        + rng.normal(0, 0.9, n)
    )

    probability = 1 / (1 + np.exp(-latent_score))
    shortlisted = (rng.random(n) < probability).astype(int)

    df = pd.DataFrame({
        "experience_years": np.round(experience, 1),
        "education_level": education,
        "relevant_skill_count": skills,
        "certification_count": certifications,
        "previous_industry": industry,
        "shortlisted": shortlisted
    })

    # Add realistic missing entries as required by the case study.
    for column, fraction in [
        ("experience_years", 0.06),
        ("certification_count", 0.05)
    ]:
        idx = rng.choice(n, int(n * fraction), replace=False)
        df.loc[idx, column] = np.nan

    return df

if __name__ == "__main__":
    out = Path("data/resume_shortlisting.csv")
    out.parent.mkdir(exist_ok=True)
    df = generate_dataset()
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} rows to {out}")
    print(df.head())
    print("\nShortlisting rate:", round(df["shortlisted"].mean() * 100, 2), "%")
