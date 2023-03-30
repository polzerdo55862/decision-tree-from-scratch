import matplotlib.pyplot as plt
import seaborn as sns
from plotly import graph_objs as go
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd

def plot_threshold_evaluation(thresholds, mse_parent_list, mse_list, threshold):
    # create figure
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=thresholds,
            y=mse_list,
            mode="lines",
            name="MSE after split",
            line=dict(color="black")
        )
    )

    fig.add_trace(
        go.Scatter(
            x=thresholds,
            y=mse_parent_list,
            mode="lines",
            name="MSE of parent node",
            line=dict(color="black", dash='dot')
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[threshold,threshold],
            y=[min(mse_list), max(mse_list)],
            mode="lines",
            name="Chosen threshold",
            line=dict(color="black", dash='dashdot')
        )
    )

    # update title in go.Figure
    fig.update_layout(title="Evaluate", yaxis_title='MSE')
    
    fig.show()

    return fig

class NodePlot():

    def __init__(self, X_parent, y_parent, threshold, selected_feature):
        self.selected_feature = selected_feature
        self.x_column = X_parent[self.selected_feature]
        self.y_parent = y_parent
        self.data_set = np.column_stack((self.x_column, y_parent))
        self.threshold = threshold

        # define a list with all observations of the left and right leaf
        self.left_y = self.data_set[self.data_set[:, 0]<self.threshold][:, 1]
        self.left_x = self.data_set[self.data_set[:, 0]<self.threshold][:, 0]
        self.right_y = self.data_set[self.data_set[:, 0]>=self.threshold][:, 1]
        self.right_x = self.data_set[self.data_set[:, 0]>=self.threshold][:, 0]

        # calculate the mean of the observations for the left and right leaf'''
        self.parent_y_mean = np.mean(self.y_parent)
        self.left_y_mean = np.mean(self.left_y)
        self.right_y_mean = np.mean(self.right_y)

        # calculate the weighted mean squared error
        self.parent_mse = np.mean((y_parent - self.parent_y_mean)**2)
        mse_l = np.mean((self.left_y - self.left_y_mean)**2)
        mse_r = np.mean((self.right_y - self.right_y_mean)**2)

        # calculate the number of instances in the parent and child nodes
        n_l = len(self.left_y)
        n_r = len(self.right_y)
        n = len(self.data_set)

        # calculate the weighted mse for child nodes
        self.child_mse = (n_l/n) * mse_l + (n_r/n) * mse_r

    def plot_split(self):
        plt.rcParams['font.size'] = '16'
        sns.set_style("darkgrid", {"axes.facecolor": ".9"})

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=self.left_x,
                y=self.left_y,
                mode="markers",
                name="Data set: left node",
                line=dict(color="grey")
            )
        )

        fig.add_trace(
            go.Scatter(
                x=self.left_x,
                y=np.linspace(self.left_y_mean, self.left_y_mean, len(self.left_x)),
                mode="lines",
                name="Right node prediction",
                line=dict(color="black")
            )
        )

        # create go.scatter plot with black line
        fig.add_trace(
            go.Scatter(
                x=self.right_x,
                y=self.right_y,
                mode="markers",
                name="Data set: right node",
                #line=dict(color="#ffe476")
                line=dict(color="black")
            )
        )

        fig.add_trace(
            go.Scatter(
                x=self.right_x,
                y=np.linspace(self.right_y_mean, self.right_y_mean, len(self.right_x)),
                mode="lines",
                name="Left node prediction",
                line=dict(color="black", dash='dot')
            )
        )

        fig.add_trace(
            go.Scatter(
                x=[self.threshold, self.threshold],
                y=[min(self.y_parent), max(self.y_parent)],
                mode="lines",
                name="MSE of parent node",
                line=dict(color="black", dash='dashdot')
            )
        )

        # update title in go.Figure
        fig.update_layout(title="Data set", xaxis_title=self.selected_feature, yaxis_title=self.y_parent.name)

        return fig.show()

class SelectedDataSet():
    '''
    Parameters
    -----------
    - chosen_dataset: options: ["auto", "boston housing"]
    '''

    def __init__(self, chosen_dataset="auto"):
        
        self.chosen_dataset = chosen_dataset

        if self.chosen_dataset == "boston housing":
            # load the boston housing data set from uci.edu
            url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data'
            cols = [
                    'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 
                    'AGE', 'DIS', 'RAD','TAX','PTRATIO','B','LSTAT','MEDV'
                    ]
            df = pd.read_csv(url, delim_whitespace=True, names=cols)

        elif self.chosen_dataset == "auto":
            # load the automobile data set from UCI.edu 
            url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data'
            df = pd.read_csv(url, header=None)

            # Name columns 
            df.columns = [
                            'symboling', 'normalized_losses', 'make', 'fuel_type', 
                            'aspiration', 'num_doors', 'body_style', 'drive_wheels', 
                            'engine_location','wheel_base','length','width','height',
                            'curb_weight','engine_type','num_cylinders','engine_size',
                            'fuel_system','bore','stroke','compression_ratio','horsepower',
                            'peak_rpm','city_mpg','highway_mpg','price'
                            ]

            df = df[(df.horsepower != '?')]
            df = df[(df.price != '?')]

            df['horsepower'] = df['horsepower'].astype(int)
            df['price'] = df['price'].astype(int)

            self.df = df

        # define the last column of the data frame as y and the rest as X
        self.y = self.df.iloc[:, -1]
        self.X = self.df.iloc[:, :-1]

def load_auto_data_set():

    # Load the automobile data set from UCI.edu 
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data'
    df = pd.read_csv(url, header=None)

    # Name columns 
    df.columns = [
                    'symboling', 'normalized_losses', 'make', 'fuel_type', 
                    'aspiration', 'num_doors', 'body_style', 'drive_wheels', 
                    'engine_location','wheel_base','length','width','height',
                    'curb_weight','engine_type','num_cylinders','engine_size',
                    'fuel_system','bore','stroke','compression_ratio','horsepower',
                    'peak_rpm','city_mpg','highway_mpg','price'
                    ]

    # Filter for lines where power and price are available
    df = df[(df.horsepower != '?')]
    df = df[(df.price != '?')]

    # Filter for lines where power and price are available
    df['horsepower'] = df['horsepower'].astype(int)
    df['price'] = df['price'].astype(int)

    # Define the last column of the data frame as y and the rest as X
    y = df.iloc[:, -1]
    X = df.iloc[:, :-1]

    return df, X, y
