import numpy as np
import pandas as pd

# Define the input_handler function


def input_handler(
    n_samples,
    n_snps,
    n_causal_snps,
    additive_variance,
    interaction_variance,
    noise_variance,
    n_pairs,
    frac_epi_additive,
):
    n_null_var = sum(
        [
            additive_variance is None,
            interaction_variance is None,
            noise_variance is None,
        ]
    )

    if n_null_var > 1:
        raise ValueError(
            "At least two of 'additive_variance', 'interaction_variance', and 'noise_variance' must be specified."
        )

    provided_variance_sum = sum(
        filter(None, [additive_variance, interaction_variance, noise_variance])
    )
    if provided_variance_sum > 1:
        raise ValueError("Variance total cannot exceed 1.")

    if additive_variance is None:
        additive_variance = 1 - provided_variance_sum
    elif interaction_variance is None:
        interaction_variance = 1 - provided_variance_sum
    elif noise_variance is None:
        noise_variance = 1 - provided_variance_sum

    if additive_variance == 0:
        n_causal_snps = 0
        frac_epi_additive = 0
    if interaction_variance == 0:
        n_pairs = 0

    assert n_samples > 0, "'n_samples' must be greater than 0."
    assert n_snps > 0, "'n_snps' must be greater than 0."
    assert n_causal_snps <= n_snps, "'n_causal_snps' can't exceed 'n_snps'."
    assert not (
        additive_variance > 0 and n_causal_snps < 1
    ), "'n_causal_snps' must > 0 if `additive_variance` is > 0."

    assert not (
        n_causal_snps > 0 and additive_variance == 0
    ), "'additive_variance' must be > 0 if `n_causal_snps` is > 0."

    assert (
        additive_variance + interaction_variance + noise_variance == 1
    ), "Variance components must sum to 1."

    assert (
        0 <= additive_variance <= 1
    ), "'additive_variance' must be within [0, 1]."

    assert (
        0 <= interaction_variance <= 1
    ), "'interaction_variance' must be within [0, 1]."

    assert 0 <= noise_variance <= 1, "'noise_variance' must be within [0, 1]."

    assert not (
        interaction_variance > 0 and n_pairs < 1
    ), "'n_pairs' must be > 0 if `interaction_variance' is > 0."

    assert not (
        n_pairs > 0 and interaction_variance == 0
    ), "'interaction_variance' must be > 0 if `n_pairs' is > 0."

    assert not (
        interaction_variance > 0 and n_pairs > n_snps / 2
    ), "'n_pairs' can't exceed n_snps/2."

    assert frac_epi_additive is None or (
        0 <= frac_epi_additive <= 1
    ), "'frac_epi_additive' must be either None or within [0, 1]."

    return (
        n_samples,
        n_snps,
        n_causal_snps,
        additive_variance,
        interaction_variance,
        noise_variance,
        n_pairs,
        frac_epi_additive,
    )


def logistic_fun(x):
    return 1 / (1 + np.exp(-x))


def standardise(array):
    return (array - np.mean(array)) / np.std(array)


def compute_contributions_vector(genotype_matrix, variance):
    n_effects = genotype_matrix.shape[1]
    rng = np.random.default_rng()
    effects = rng.standard_normal(n_effects)
    contributions = np.dot(genotype_matrix, effects)

    # Rescaling
    contributions *= np.sqrt(variance / np.var(contributions))

    return contributions


def sim_genmatrix(
    n_samples=100,
    n_snps=10,
    n_causal_snps=2,
    additive_variance=None,
    interaction_variance=None,
    noise_variance=None,
    n_pairs=1,
    frac_epi_additive=None,
    maf=0.5,
):
    # Validate inputs
    (
        n_samples,
        n_snps,
        n_causal_snps,
        additive_variance,
        interaction_variance,
        noise_variance,
        n_pairs,
        frac_epi_additive,
    ) = input_handler(
        n_samples,
        n_snps,
        n_causal_snps,
        additive_variance,
        interaction_variance,
        noise_variance,
        n_pairs,
        frac_epi_additive,
    )

    data = np.random.binomial(2, maf, (n_samples, n_snps))

    if additive_variance > 0:
        causal_snp_inds = np.random.choice(n_snps, n_causal_snps, replace=False)
        causal_snps = data[:, causal_snp_inds]
        additive_contributions = compute_contributions_vector(
            causal_snps, additive_variance
        )
    else:
        additive_contributions = np.zeros(n_samples)

    if interaction_variance > 0:
        if frac_epi_additive is None:
            frac_epi_additive = n_causal_snps / n_snps

        non_causal_inds = np.setdiff1d(np.arange(n_snps), causal_snp_inds)
        n_epistatic = n_pairs * 2
        n_additive_epistatic = round(n_epistatic * frac_epi_additive)
        n_nonadditive_epistatic = n_epistatic - n_additive_epistatic

        additive_epistatic_inds = np.random.choice(
            causal_snp_inds, n_additive_epistatic, replace=False
        )

        nonadditive_epistatic_inds = np.random.choice(
            non_causal_inds, n_nonadditive_epistatic, replace=False
        )

        all_epistatic_inds = np.concatenate(
            [additive_epistatic_inds, nonadditive_epistatic_inds]
        )

        epistatic_pairs = []
        while len(all_epistatic_inds) > 0:
            pair = np.random.choice(all_epistatic_inds, 2, replace=False)
            epistatic_pairs.append(pair)
            all_epistatic_inds = np.setdiff1d(all_epistatic_inds, pair)

        pairs_array = np.array(epistatic_pairs)
        interaction_matrix = (
            data[:, pairs_array[:, 0]] * data[:, pairs_array[:, 1]]
        )

        epistatic_contributions = compute_contributions_vector(
            interaction_matrix, interaction_variance
        )

    else:
        epistatic_contributions = np.zeros(n_samples)

    noise = np.random.normal(0, np.sqrt(noise_variance), n_samples)
    phenotype = additive_contributions + epistatic_contributions + noise

    data = np.column_stack((data, phenotype))

    return data


def model_simulation_generic(
    model_generator,
    n_samples,
    n_snps,
    n_causal_snps,
    additive_variance,
    interaction_variance,
    n_pairs,
    frac_epi_additive=None,
    maf=0.5,
    n_reps=10,
):
    r_squared_vec = np.zeros(n_reps)
    adj_r_squared_vec = np.zeros(n_reps)
    for i in range(n_reps):
        array = sim_genmatrix(
            n_samples=n_samples,
            n_snps=n_snps,
            n_causal_snps=n_causal_snps,
            additive_variance=additive_variance,
            interaction_variance=interaction_variance,
            noise_variance=1 - additive_variance - interaction_variance,
            n_pairs=n_pairs,
            frac_epi_additive=frac_epi_additive,
            maf=maf,
        )

        data = pd.DataFrame(
            array, columns=[f"SNP_{j+1}" for j in range(n_snps)] + ["phenotype"]
        )

        sample_size = len(data)
        train_inds = np.random.choice(
            range(sample_size), size=sample_size // 2, replace=False
        )

        train_data = data.iloc[train_inds]
        test_data = data.iloc[~data.index.isin(train_inds)]

        model = model_generator()
        model.fit(
            train_data.drop(columns=["phenotype"]), train_data["phenotype"]
        )

        predictions = model.predict(test_data.drop(columns=["phenotype"]))

        residuals = test_data["phenotype"] - predictions

        ss_res = np.sum(residuals**2)
        ss_tot = np.sum(
            (test_data["phenotype"] - np.mean(test_data["phenotype"])) ** 2
        )

        r_squared = 1 - (ss_res / ss_tot)
        adj_r_squared = 1 - (1 - r_squared) * (
            (len(test_data) - 1) / (len(test_data) - len(test_data.columns) + 1)
        )

        r_squared_vec[i] = r_squared
        adj_r_squared_vec[i] = adj_r_squared

    total_variance = additive_variance + interaction_variance
    results = pd.DataFrame(
        {
            "additive_variance": [additive_variance] * n_reps,
            "interaction_variance": [interaction_variance] * n_reps,
            "frac_epi_additive": [frac_epi_additive] * n_reps,
            "r_squared": r_squared_vec,
            "adj_r_squared": adj_r_squared_vec,
            "true_total_variance": [total_variance] * n_reps,
        }
    )

    return results
