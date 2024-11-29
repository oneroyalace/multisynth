import random
from collections.abc import Iterable
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import minimize
from numpy.typing import NDArray
from tqdm import tqdm


class MultiSynth:
    def __init__(self):
        self._lambda_val = None
        self._synthetic_analyses = None
        self._indexed_treated_effects = None
        self._indexed_placebo_effects = None

    """
    Constructs a dataframe of unit covariates for matching by selecting, averaging, and normalizing variables from the input data panel as specified in `variable_specs` 
    args:
        data (pd.DataFrame): A balanced panel containing unit variables
        variable_spects (list[dict]: A list of dictionaries describing how new variables for synthetic control matching should be constructed
        treatment_period (int): The period in which treatment occurs for the unit we're constructing a synthetic control for
        treatment_period_index (int): `treatment_period` offset from the earliest date in the panel 
        unit_var (str): The name of the column in the panel dataframe containing unit identifiers
        time_var (str): The name of the column in the panel dataframe containig time identifiers
    returns: 
        A dataframe with covariates constructed for each unit as specified in `variable_specs` specification. Columns are units, rows are covariates
    """
    def construct_variables_for_unit_matching(self, 
                                              data: pd.DataFrame, 
                                              variable_specs: list[dict], 
                                              treatment_period: int, 
                                              treatment_period_index: int, 
                                              unit_var: str, 
                                              time_var: str) -> pd.DataFrame:
        constructed_vars = []

        for variable_spec in variable_specs:
            var = variable_spec["var"]
            specification = variable_spec["specification"]
            
            agg_option = specification["aggregate_option"]
            time_periods = specification.get("periods")
            should_normalize = specification.get("normalize")
            
            df = data.copy()
            time_periods = time_periods or range(int(min(df[time_var])), int(treatment_period))  # If time period selection isn't provided, use avg until treatment

            # Normalize curent current variable w.r.t. its value in the period just before treatment cocurs
            if should_normalize:
                var_tminus1 = df.groupby(unit_var)[[unit_var, var]].nth(treatment_period_index-1).rename(columns={var: f"{var}_t-1"})  # contains t-1 value for `var` for each unit
                # print(df)
                # print(var_tminus1)
                df = df.merge(var_tminus1, on="cty_fips")
                df[var] = df[var] / df[f"{var}_t-1"]  # perform normalization
                
            # If user specifed that a variable should be averaged over a set of periods to construct a covariate for matching...
            if agg_option == "avg":
                # construct a df containing values of the averaged variable for each unit
                new_var_df = df[df[time_var].isin(time_periods)].groupby(unit_var)[var].mean().to_frame().reset_index()  # group df by unit, then avg {var} over periods specified in `variable_spec`
                new_var_df["var"] = var + f"_avg_{time_periods[0]}_{time_periods[-1]}"  # set "var" column to variable name (for reshaping below)  
                new_var_df = new_var_df.rename(columns = {var: "value"})  # rename column for reshaping below
                constructed_vars.append(new_var_df)
            # If user specified that a variable's value in a specific (set of) years should be used for matching...
            elif agg_option == "individual":
                for period in time_periods:
                    new_var_df = df[df[time_var] == period][[unit_var, var]]  # Select values of `var` for given period
                    new_var_df["var"] = f"{var}_{period}"  # set "var" column to variable name (for reshaping below)
                    new_var_df = new_var_df.rename(columns = {var: "value"})  # rename column for reshaping below
                    constructed_vars.append(new_var_df)
        return pd.concat(constructed_vars).pivot(index="var", columns=unit_var, values="value")  # reshape dataframe

    """
    Main access point to this class. Fits a synthetic control model to the input data and returns resulting data

    args:
        data (pd.DataFrame): A balanced panel containing unit variables
        variable_spects (list[dict]): A list of dictionaries describing how new variables for synthetic control matching should be constructed
        treatment_var (str): The name of the column in the panel dataframe denoting when the given unit is treated
        unit_var (str): The name of the column in the panel dataframe containing unit identifiers
        time_var (str): The name of the column in the panel dataframe containig time identifiers
        outcome_var (str): The name of the colum in the panel dataframe containing the outcome variable
        penalized (bool): If true, estimate a penalized/bias-controlled synthetic control model. Otherwise, estimate the base model 
        normalize_outcomes (bool): If true, report percent rather than level changes in the outcome variable 
    returns:
        A dataframe that for each treated unit contains average treatment effects, rmspes for treated and placebo units, p values,
        unit weights, treatment effects, covariate balances, treated outcomes, and placebo outcomes
    """
    def fit_multisynth(self, 
                       data: pd.DataFrame, 
                       variable_specs: list[dict], 
                       treatment_var: str, 
                       unit_var: str, 
                       time_var: str, 
                       outcome_var: str, 
                       penalized: bool =False,
                       normalize_outcomes: bool =True,
                       verbose: bool =False) -> pd.DataFrame:
        
        treated_rows = data.groupby(unit_var).filter(lambda group: not group[treatment_var].isna().all())
        treated_units = set(treated_rows[unit_var])                                                       
        never_treated_units = list(set(data[unit_var]).difference(treated_units))

        data = data.sort_values([unit_var, time_var])
        weights = None
        synthetic_fits = []
        for unit_ind, treated_unit in enumerate(treated_units):
            print(f"Estimating synthetic control for unit {unit_var}={treated_unit} ({unit_ind+1}/{len(treated_units)})")
            treatment_period = data[data[unit_var] == treated_unit].iloc[0][treatment_var]  # period when treatment is first administered
            

            # number of periods until treatment occurs. If treatment occurs in 1965 and panel begins in 1961,
            # treatment occurs in the 5th period (0-index = 4)
            # so post-treatment begins in index 5
            # This assumes balanced panel
            treatment_period_index = int(treatment_period - min(data[time_var]))

            outcomes = data[[unit_var, time_var, outcome_var]].pivot(columns=unit_var, index=time_var, values=outcome_var)
            if normalize_outcomes:
                outcomes = outcomes.apply(lambda col: col/col.iloc[treatment_period_index-1], axis=0)  # Normalize to t-1 period
            control_outcomes = outcomes[never_treated_units]
            
            constructed_variables = self.construct_variables_for_unit_matching(data=data, 
                                                                               variable_specs=variable_specs,
                                                                               treatment_period=treatment_period,
                                                                               treatment_period_index=treatment_period_index,
                                                                               unit_var=unit_var, 
                                                                               time_var=time_var)
            
            treated_covariates = constructed_variables[[treated_unit]]
            control_covariates = constructed_variables[never_treated_units]
            

            # Find optimal weights of control units to match treated unit
            weights = self.fit_synthetic(treated_covariates=treated_covariates, 
                                         control_covariates=control_covariates, 
                                         control_outcomes=control_outcomes,
                                         unit_var=unit_var, 
                                         time_var=time_var,
                                         outcome_var=outcome_var,
                                         treatment_period_index=treatment_period_index,
                                         penalized=penalized)
            unit_weights = pd.DataFrame({"unit": never_treated_units, "weight": weights})

            synthetic_covariates = np.dot(control_covariates.to_numpy(), weights)
            covariate_balances = treated_covariates.copy()
            covariate_balances[f"synth_{treated_unit}"] = synthetic_covariates
            
            
            treatment_effects = self.calculate_treatment_effects(outcomes, 
                                                                 treated_unit=treated_unit,
                                                                 control_units=never_treated_units,
                                                                 weights=weights)
            
            # Find average treatment effect 
            ate = self.calculate_ate(treatment_effects=treatment_effects,
                                     treatment_period_index=treatment_period_index)
            
            # Find RMSPE using treated county 
            rmspe_treatment = self.calculate_rmspe(treatment_effects=treatment_effects, 
                                                   treatment_period_index=treatment_period_index)


            placebo_ATEs, placebo_RMSPEs, placebo_treatment_effects, placebo_outcomes, placebo_units = self.estimate_placebos(control_units=never_treated_units,
                                                                                                                              treated_unit=treated_unit,
                                                                                             constructed_variables=constructed_variables,
                                                                                             outcomes=outcomes,
                                                                                             unit_var=unit_var,
                                                                                             time_var=time_var,
                                                                                             outcome_var=outcome_var,
                                                                                             treatment_period_index=treatment_period_index,
                                                                                             penalized=penalized)
            p_val_rmspe = np.sum((placebo_RMSPEs > rmspe_treatment)/len(placebo_RMSPEs))
            treated_outcomes = list(outcomes[treated_unit])
            synthetic_fits.append({
                "treated_unit": treated_unit,
                "treatment_period_index": treatment_period_index,
                "ate": ate,
                "rmspe_treated": rmspe_treatment,
                "rmspe_placebo": placebo_RMSPEs,
                "p_rmspe": p_val_rmspe,
                "unit_weights": unit_weights,
                "treatment_effects": treatment_effects,
                "placebo_unit": placebo_units,
                "placebo_ATEs": placebo_ATEs,
                "covariate_balances": covariate_balances,
                "treated_outcomes": treated_outcomes,
                "placebo_treatment_effects": placebo_treatment_effects,
                "placebo_outcomes": placebo_outcomes
            })
         
            self._lambda_val=None
            
        synthetic_fits = pd.DataFrame(synthetic_fits).set_index("treated_unit")
        self._synthetic_analyses = synthetic_fits
        self.produce_indexed_treatment_and_placebo_effects()
        return synthetic_fits


    """
    Calculate p-value by comparing RMSPEs of treated units to those from randomly selected placebos
    """
    def get_p_value(self) -> float:
        placebo_gt_treat_rmspe = 0
        for _ in range(1000):
            analysis = self._synthetic_analyses.sample(n=1).iloc[0]
            placebo_rmspe = random.choice(analysis["rmspe_placebo"])
            treatment_rmspe = analysis["rmspe_treated"]
            if placebo_rmspe > treatment_rmspe:
                placebo_gt_treat_rmspe += 1
        return placebo_gt_treat_rmspe / 1000
    

    """
    Given an an array of control units, fit synthetic control models for each, assuming treatment occurs at period `treatment_period_index`
    params:
        control_units: A list of control unit names that identify columns in the `outcomes` dataframe
        treated_unit: The name of the treated unit
        constructed_variables: A dataframe of covariates used to build  synthetic matches for treated units
        outcomes: A dataframe consisting of outcomes for all units over the observation period
        unit_var: The name of the variable that identifies a unit (e.g "state")
        time_var: The name of the variable that identifies time periods (e.g. "year")
        outcome_var: The name of the outcome variable
        treatment_period_index: The index of the period in which treatment occurs. Calculated relative to the first date in the observation period
    returns:
        [
            1) an array of average treatment effects using each control unit as a placebo 
            2) an array of RMSPEs calcluated using each control unit as a placebo
            3) a num_placebos x num_time_periods array of treatment effects over time using each control unit as a placebo
            4) a num_placebos x num_time_periods array of placebo outcomes
            5) an array of placebo unit names
        ]       
    """
    def estimate_placebos(self, 
                          control_units: list[str|int] , 
                          treated_unit,
                          constructed_variables: pd.DataFrame, 
                          outcomes: pd.DataFrame,
                          unit_var: str, 
                          time_var: str, 
                          outcome_var: str, 
                          treatment_period_index: int,
                          penalized=False) -> Tuple[NDArray[float], NDArray[float], NDArray[NDArray[float]], NDArray[NDArray[float]]]:

        placebo_ATEs = []
        placebo_RMSPEs = []
        placebo_treatment_effect_arr = []
        all_placebo_outcomes = []
        placebo_units = []

        control_units = sorted(control_units)
        
        for placebo in control_units:
            # print("placebo", placebo)
            placebo_donors = [donor for donor in control_units if donor != placebo]
            placebo_donors.append(treated_unit)
            placebo_covariates = constructed_variables[[placebo]]
            placebo_donor_covariates = constructed_variables[placebo_donors]   
            placebo_donor_outcomes = outcomes[placebo_donors]
            # print(placebo_covariates)
            # print(placebo_donor_covariates)
            # print(placebo_donor_outcomes)
            # exit()

            weights = self.fit_synthetic(treated_covariates=placebo_covariates, 
                                         control_covariates=placebo_donor_covariates,
                                         control_outcomes=placebo_donor_outcomes,
                                          unit_var=unit_var, 
                                          time_var=time_var, 
                                          outcome_var=outcome_var,
                                          treatment_period_index=treatment_period_index,
                                          penalized=penalized)
            
            placebo_outcomes = self.calculate_synthetic_outcomes(outcomes=outcomes,
                                                                 control_units=placebo_donors,
                                                                 weights=weights)
            all_placebo_outcomes.append(placebo_outcomes)
            
            placebo_treatment_effects = self.calculate_treatment_effects(outcomes, 
                                                                         treated_unit=placebo,
                                                                         control_units=placebo_donors,
                                                                         weights=weights)
            
            placebo_treatment_effect_arr.append(placebo_treatment_effects)
            placebo_ate = self.calculate_ate(treatment_effects=placebo_treatment_effects, 
                                             treatment_period_index=treatment_period_index)
            placebo_ATEs.append(placebo_ate)

            placebo_rmspe = self.calculate_rmspe(treatment_effects=placebo_treatment_effects, 
                                                 treatment_period_index=treatment_period_index)
            placebo_RMSPEs.append(placebo_rmspe)
            placebo_units.append(placebo)

        return (np.array(placebo_ATEs), 
                np.array(placebo_RMSPEs), 
                np.array(placebo_treatment_effect_arr),
                np.array(all_placebo_outcomes),
                np.array(placebo_units))


    """
    Calculate outcomes for synthetic control unit by multiplying control outcomes by weight vector
    args:
        control_units (list[str|int]): A list of identifiers for control units
        weights (np.array: A vector of unit weights
        outcomes (pd.DataFrame): A list of outcomes for each time period and unit

    """
    def calculate_synthetic_outcomes(self, 
                                     outcomes: pd.DataFrame, 
                                     control_units: list[str|int], 
                                     weights: np.array):
        X_1 = outcomes[control_units].to_numpy()
        Y_star = np.dot(X_1, weights)
        return Y_star
        
    def calculate_treatment_effects(self, outcomes: pd.DataFrame, treated_unit: str, control_units, weights: np.array):
        Y_star = self.calculate_synthetic_outcomes(outcomes=outcomes,
                                                   control_units=control_units,
                                                   weights=weights)
        Y_1 = outcomes[[treated_unit]].to_numpy().reshape(-1)
        treatment_effects = Y_1 - Y_star
        return treatment_effects


    """
    Constructs a dictionary with effect values labeled by their occurence relative in time to the treatment period
    Given a treatment effect array [0,1,3,2,5] and 0-indexed treatment_period=2, return 
    {
        -2: 0,
        -1: 1,
         0: 3,
         1: 2,
         2: 5
    }
    
    args: 
        arr: (Iterable[list, np.array] An iterable to be re-indexed
        treatment_period: the period used to index arr around. 
            If treatment_period = x, the xth value in arr will be indexed with time t=0
                                     the x-1th value in arr will be indexed with time t=-1
                                     the x+1th value in arr will be indexed with time t=1
    returns:
        A dictionary with keys equal to array positions relative to the intput treatment_period and values equal to array values
    """
    def index_array_around_treatment_time(self, arr: Iterable, treatment_period_index: int) -> dict[int, float]:
        period_indexed_effects = {}
        unit_weight = 1
        for ind, value in enumerate(arr):
            periods_since_treat = ind - treatment_period_index
            period_indexed_effects[periods_since_treat] = value #* unit_weight
        return period_indexed_effects

    
    """
    Create time period-indexed dataframes of treatment effects for each treated unit and treated placebo. Indices are centered around treatment times
    
    args:
        --
    returns:
        (pd.DataFrame, pd.DataFrame): Two dataframes, both with period offsets as columns. One df's rows are treatment unit effects, others are placebo effects 
    """
    def produce_indexed_treatment_and_placebo_effects(self):
        treated_unit_effects = []
        placebo_effects = []
        for treated_unit, analysis in self._synthetic_analyses.iterrows():
            treatment_effects = analysis["treatment_effects"]
            treatment_period_index = analysis["treatment_period_index"]
            
            year_indexed_effects = self.index_array_around_treatment_time(treatment_effects, treatment_period_index)  # convert treatment effect array into time-indexed dictionary format
            year_indexed_effects = year_indexed_effects | {"treated_unit": treated_unit}  #  label year-indexed treated  effects w/ treated unit
            treated_unit_effects.append(year_indexed_effects)
        
            # I need to weight the placebos.... but i'm not sure i report the names of  placebos units above
            
        
            # We should n_control_units - 1  placebos for every control unit
            for current_treatment_placebo_effects in analysis["placebo_treatment_effects"]: 
                year_indexed_placebo_effect = self.index_array_around_treatment_time(current_treatment_placebo_effects, treatment_period_index)
                year_indexed_placebo_effect = year_indexed_placebo_effect | {"treated_unit": treated_unit}  #  label year-indexed placebo effects w/ treated unit
                placebo_effects.append(year_indexed_placebo_effect)
            
        # print("Only using treatment/placebo effects from a single experiment right now")
        placebo_effects_df = pd.DataFrame(placebo_effects).set_index("treated_unit")
        treated_unit_effects_df = pd.DataFrame(treated_unit_effects).set_index("treated_unit")
        self._indexed_treated_effects = treated_unit_effects_df
        self._indexed_placebo_effects = placebo_effects_df
        return treated_unit_effects_df, placebo_effects_df


    """
    Derive `num_placebos` placebo average effects by sampling placebos from various treated units

    args:
        placebo_effects_df (pandas.DataFrame): Dataframe with row indices as units and time periods relative to treatment as columns
        num_placebos (int): The number of placebo effect paths to derive
    returns:
        A dataframe of time-indexed placebo effect paths
    """
    def sample_and_average_placebo_effects(self, placebo_effects_df: pd.DataFrame, num_placebos: int) -> pd.DataFrame:
        sample_avgs = []
        treated_units = placebo_effects_df.index.unique()  # index is set to treated units
        for _ in range(num_placebos):
            sampled_placebo_effects = placebo_effects_df.groupby("treated_unit").sample(n=1)  # select a rnadom  placebo effect estimate associated with each treated unit 
            avg_sampled_placebo_effects = sampled_placebo_effects.mean()  # average the sampled placebo effects
            sample_avgs.append(avg_sampled_placebo_effects)
        return pd.DataFrame(sample_avgs)
        

    """
    Plot treatment and placebo paths

    args:
        num_placebos (int): The number of placebo paths to plot
        pre_treat_periods (int): The number of periods before treatment to show in the plot
        post_treat_periods (int): The number of periods after treatment to show in the plot
        ylim (tuple): The y bounds for the plot. By default, let seaborn set these. 
    returns:
        None
    """
    def plot_treatment_and_placebos(self,
                                    num_placebos: int = 100,
                                    pre_treat_periods: int = 5,
                                    post_treat_periods: int = 5,
                                    ylim: tuple = None) -> None:

        indexed_treated_effects =  self._indexed_treated_effects.mean()

        
        if len(self._synthetic_analyses) == 1:  # if there's only a single treated unit, we don't need to sample placebos. We can just use the placebos from the treated unit
            sample_placebo_avgs = self._indexed_placebo_effects
        else:  # Otherwise, sample placebos from each treated unit and average them
            sample_placebo_avgs = self.sample_and_average_placebo_effects(self._indexed_placebo_effects, num_placebos)


        
        plt.figure(figsize=(8,5))
        for _, row in sample_placebo_avgs.iterrows():
            sns.lineplot(x=row.index, y = row.values, color="grey")  # indices are periods relative to treatment. 
        sns.lineplot(x=indexed_treated_effects.index, y=indexed_treated_effects.values, color="green")
    
        plt.xlim(-1*pre_treat_periods, post_treat_periods)
        if ylim: 
            plt.ylim(ylim)
        plt.axvline(x=0, color="black", linestyle="--", linewidth=1)
        plt.xticks(ticks=range(-1 * pre_treat_periods, post_treat_periods+1), rotation=90 )
        
        ax = plt.gca()
        ax.set_xlabel("Treatment period")
        ax.set_ylabel("Normalized avg treatment effect")
        ax.set_title("Normalized treatment and placebo effects")


        plt.legend(
            handles=[
                plt.Line2D([], [], color="grey", label="Placebo"),
                plt.Line2D([], [], color="green", label="Treated Unit Avg"),
            ],
            loc="upper right",  # Legend position
        )
        plt.show()        

    
    """
    Return covariate comparisons between a single treated unit and its synthetic imitation
    args:
        treated_unit (str): The unit identifier to return covariate comparisons for
    returns:
        A pandas dataframe with unit columns and covariate rows
    """
    def get_synthetic_covariate_comparisons(self, treated_unit: str|int):
        return self._synthetic_analyses.loc[treated_unit]["covariate_balances"]


    """
    Calculate the average of treatment effects over all treated units and `post_treat_periods` tiem periods
    args:
        post_treat_periods (int): The number of periods after treatment to average effects over
    returns:
        (float) The average
    """
    def get_ate(self, post_treat_periods: int = None) -> float:
            post_treat_columns =  self._indexed_treated_effects.loc[:, self._indexed_treated_effects.columns >= 0]
            return self.calculate_ate(treatment_effects = post_treat_columns.mean(),
                                      treatment_period_index = 0,
                                      post_treat_periods=post_treat_periods)


    """
    Return control unit weights for the synthetic version of `treated_unit`
    args:
        treated_unit (int|str): The number of periods after treatment to average effects over
    returns:
        (pd.Dataframe) unit weights
    """
    def get_unit_weights(self, treated_unit: str|int) -> pd.DataFrame:
        unit_weights =  self._synthetic_analyses.loc[treated_unit]["unit_weights"]
        return unit_weights[unit_weights["weight"] > 1e-3]


    """
    Calculate average treatment effect over `post_treat_periods` periods

    args:
        treatment_effects (Iterable): a vector or list of treatment efects to be averaged post-greatment
        treatment_period_index (int): denotes the index after which treatment_effects should be averaged
        post_treat_periods (int): denotes the number of periods after treatment that should be averaged
    returns:
        A float average
    """
    def calculate_ate(self,
                      treatment_effects: Iterable[float], 
                      treatment_period_index: int = None,
                      post_treat_periods: int = None) -> float:

        # By default, calculate ATE over all periods after treatment
        if post_treat_periods is None:
            post_treat_periods = treatment_effects.shape[-1] - treatment_period_index
        post_treat_effects = treatment_effects[treatment_period_index:post_treat_periods+1]
            
        ate = np.sum(post_treat_effects)/post_treat_periods
        return ate
        
    """
    Calculate RMSPE for a treated path (Wiltshire 2023)
    args:
        treatment_effects (Iterable[float]): A list of treatment effect values
        treatment_period_index (int): The index of `treatment_effects` at which treatment occurs
    returns:
        (float) RMSPE
    """
    def calculate_rmspe(self, 
                        treatment_effects: Iterable[float],
                        treatment_period_index: int) -> float:
        
        post_treat_periods = treatment_effects.shape[-1] - treatment_period_index
        pre_treat_periods = treatment_period_index
        
        pre_treat_gaps = treatment_effects[:treatment_period_index]
        post_treat_gaps = treatment_effects[treatment_period_index:]
        
        rmspe = np.sum(np.square(post_treat_gaps)/post_treat_periods) / np.sum(np.square(pre_treat_gaps)/pre_treat_periods)
        return rmspe


    """
    Call methods to calculate synthetic control weights. If using penalized synthetic control, calculate lambda first.

    args:
        treated_covariates (pd.DataFrame): A dataframe of treated unit covariates. Row indices are covariates
        control_covariates: (pd.DataFrame): A dataframe of control unit covariates. Row indices are covariates, columns are control units
        control_outcomes (pd.DataFrame): A dataframe of outcomes for control units. Row indices are time periods, columns are control units
        unit_var (str): The name of the variable that identifies a unit (e.g "state")
        time_var (str): The name of the variable that identifies time periods (e.g. "year")
        outcome_var (str): The name of the outcome variable
        treatment_period_index (int): The index of the period in which treatment occurs. Calculated relative to the first date in the observation period
        penalized (bool): If true, use penalized synthetic control
    returns:
        (np.array[float]) unit weights
    """
    def fit_synthetic(self, 
                      treated_covariates: pd.DataFrame, 
                      control_covariates: pd.DataFrame, 
                      control_outcomes: pd.DataFrame,
                      unit_var, 
                      time_var, 
                      outcome_var, 
                      treatment_period_index: int, 
                      penalized=False) -> NDArray[float]:

        X_0 = control_covariates.to_numpy()
        X_1 = control_outcomes.to_numpy()
        Y_0 = treated_covariates.to_numpy()
        if penalized == True:
            if self._lambda_val is None:
                self._lambda_val = self.find_optimal_lambda_by_leave_one_out_cv(X_0=X_0, 
                                                                          X_1=X_1, 
                                                                          treatment_period_index=treatment_period_index)
            weights = self.calculate_weights_penalized_synthetic(X_0=X_0, 
                                                                 X_1=X_1,
                                                                 Y_0=Y_0, 
                                                                 treatment_period_index=treatment_period_index)
        elif penalized == False:
            weights = self.calculate_weights_base_synthetic(X_0, Y_0)
        return weights
        
    """
    Estimate weights for base Synthetic Control as described in Abadie & Gardeazabal (2003)
    """
    def calculate_weights_base_synthetic(self, X_0, Y_0):
        num_covariates = X_0.shape[0]  # Rows in X_0 represent individual covariates
        cov_weights = [1/num_covariates] * num_covariates  # weight all covariates equally for now. Abadie suggests using regressions to determine weights
        V = np.diag(cov_weights)

        def objective(W):
            # the residuals Y_0 - X_0 W
            W = np.reshape(W, (num_control_units, 1))
            residual = Y_0 - np.dot(X_0, W)
            penalty = np.dot(residual.T, np.dot(V, residual)).item()
            return float(penalty)
        
        
        # Find weights that minimize (X_1 - X_0 W)'V(X_1 - X_0 W) s.t. entries of w are positive
        num_control_units = X_0.shape[1]
        
        bounds = [(0,1) for _ in range(num_control_units)]  # weights must be non-negative 
        
        
        constraints = {"type": "eq", "fun": lambda W: np.sum(W) - 1}
        naive_weights = np.full(num_control_units, 1/num_control_units)
        result = minimize(objective,
                          x0=naive_weights,
                          bounds=bounds,
                          constraints = constraints,
                          method="SLSQP")
        optimal_weights = result.x 
        
        return optimal_weights

    
    """
    Estimate weights for penalized synthetic control as described in Abadie and L'Hour 2021.
    Given J covariates for matching, K control units, and L time periods
    
    args:
        X_0 (np.NDArray): A 2-D numpy array with control unit columns and covariate rows. Dimensions: (J X K)
        X_1 (np.NDArray): A 2-D numpy array with control unit columns and rows for the outcome variable in each time period. Dimensions: (L X K)
        Y_0 (np.NDArray): A 1-D numpy array with covariate values for the treated unit. Dimensions: (J X 1)
        treatment_period_index (int): The index of the period (offset from the panel start period) in which the treated unit receives treatment

    Returns: 
        np.NDArray: A 1-D numpy array with optimal weightings of control units to create a synthetic control. Dimensions: (K X 1)
    """
    def calculate_weights_penalized_synthetic(self, 
                                              X_0: NDArray[NDArray[float]], 
                                              X_1: NDArray[NDArray[float]],
                                              Y_0: NDArray[NDArray[float]],
                                              treatment_period_index: int) -> NDArray[float]:

        num_control_units = X_0.shape[1]
        naive_weights = np.full(num_control_units, 1/num_control_units)

        pairwise_diffs = np.subtract(X_0, Y_0.reshape(-1,1))
        pairwise_dists = np.diag(pairwise_diffs.T @ pairwise_diffs)

        # Set up QP variables
        P = X_0.T @ X_0
        q = -1.0 * Y_0.T @ X_0 + (self._lambda_val/2.0) * pairwise_dists.T
        q = q.squeeze()

        # Solve QP problem
        def objective(W):
            return  0.5 * W.T @  P @ W + q.T @ W

        # weights are non-negative and sum to 1
        constraints = [{'type': 'eq', 'fun': lambda W: np.sum(W) - 1}]
        bounds = [(0, None) for _ in range(num_control_units)]
        result = minimize(objective, 
                          x0=naive_weights, 
                          constraints=constraints, 
                          bounds=bounds)
        
        optimal_weights = result.x
        return optimal_weights

    """
    Find the optimal penalization coefficient lambda by minimizing squared prediction error for control units (Abadie and L'Hour 2021)

    args:
        X_0 (np.NDArray): A 2-D numpy array with control unit columns and covariate rows. Dimensions: (J X K)
        X_1 (np.NDArray): A 2-D numpy array with control unit columns and rows for the outcome variable in each time period. Dimensions: (L X K)
    returns:
        (float): A lambda value between [0,1] that minimizes squared prediction errors
    """
    def find_optimal_lambda_by_leave_one_out_cv(self, 
                                                X_0, 
                                                X_1, 
                                                treatment_period_index: int):

        lambdas = np.linspace(0.01, 1, 15)  
        num_control_units = X_0.shape[1]
        
        best_lambda = None
        min_loss = float("inf")
        
        for lambda_val in lambdas:
            # print(f"Calculating loss, λ={lambda_val}")
            self._lambda_val = lambda_val
            total_loss = 0
            for i in range(num_control_units):  # iterate through control units
                placebo_covs = X_0[:, i]   
                donor_pool_covs = np.delete(X_0, i, axis=1)
                donor_pool_outcomes = np.delete(X_1, i, axis=1)
            
                weights = self.calculate_weights_penalized_synthetic(X_0=donor_pool_covs, 
                                                                     X_1=donor_pool_outcomes,
                                                                     Y_0=placebo_covs,  
                                                                     treatment_period_index=treatment_period_index)
                
                post_intervention = np.arange(treatment_period_index, X_1.shape[0])  # post-intervention time periods
                synthetic_outcomes = np.dot(donor_pool_outcomes[post_intervention, :], weights)
                actual_outcomes = X_1[post_intervention, i]
            
                total_loss += np.sum((actual_outcomes - synthetic_outcomes) ** 2)  # add squared prediction error to total loss
            if total_loss < min_loss:
                min_loss = total_loss
                best_lambda = lambda_val
        return best_lambda

