import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt 
import folium
from streamlit_folium import folium_static

# Main Streamlit app code
# Title of the app
st.header('Junyi Academy Learning Activity Analysis', divider='rainbow')

st.title('Students')
st.write("Mohammad Mahdi Heydari Asl (D03000037)")
st.write("Seyed Sadegh Elmi Mousavi (D0300009)")
st.write("Seyedeh Sara Hashemi (D03000036)")

# Section 1: Dataset Descriptions and Statistics
st.header('Info_UserData.csv')
st.write("""
Describes the metadata of the selected registered students in Junyi Academy
         """)

uhead = pd.read_csv("userhead.csv")

# Display the DataFrame
st.dataframe(uhead)


# Description for df_problem
st.subheader('Log_Problem.csv')
st.write("""
Recorded 16,217,311 problem attempt logs of 72,758 students for a year from 2018/08/01 to 2019/07/31.
""")

lhead = pd.read_csv("df_head.csv")

# Display the DataFrame
st.dataframe(lhead)

st.write("""
#### Observing the table, we can conclude that:

1. "problem number" is the problem order in the exercise for the user

2. "exercise_problem_repeat_session" is how many times the user encounter this problem

3. The timestamp is rounded to the nearest 15 minute interval to preserve privacy in the dataset
         """)


# Description for df_content
st.subheader('Info_Content.csv')
st.write("""
Describes the metadata of the exercises, each exercise is a basic unit of learning consisted of many problems.
""")

chead = pd.read_csv("contenthead.csv")

# Display the DataFrame
st.dataframe(chead)

st.title('Number of users per gender')
st.image('Gender.png')
st.write("""
The Majority of the students **do not set** their gender on the platform.
""")

st.header('Top 5 city all of users')
st.write("""
| City        | tp           | ntpc      |tc |ty |kh |
| ------------- |:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
| User      | 12494 |10808      |10710      |7615      |6888     |

         """)



# Coordinates for the center of Taiwan
taiwan_center = [23.6978, 120.9605]

# Create the map centered around Taiwan
taiwan_map = folium.Map(location=taiwan_center, zoom_start=7)

# Define the cities and their coordinates
cities = [
    {"name": "Taipei", "coordinates": [25.0330, 121.5654]},
    {"name": "New Taipei", "coordinates": [25.0169, 121.4628]},
    {"name": "Taichung", "coordinates": [24.1477, 120.6736]},
    {"name": "Taoyuan", "coordinates": [24.9936, 121.3010]},
    {"name": "Kaohsiung", "coordinates": [22.6273, 120.3014]}
]

# Add markers for each city
for city in cities:
    folium.Marker(
        location=city["coordinates"],
        popup=city["name"],
        icon=folium.Icon(color='blue', icon='info-sign')
    ).add_to(taiwan_map)

# Display the map in Streamlit
folium_static(taiwan_map)

st.header('The relation between levels, exercise and problem')
st.image('level.jpg')

st.header('Impact of Academic Calendar and Holidays on User Login Trends in junyi academy')
st.image('firstlogin.png')
st.write("""
#### Reference the first login date (first_login_date_TW).
We observed a trend showing a surge at the beginning of the semester and the start of holidays. 
The highest activity occurred on August 31, 2018, the second day of the fall semester.
The lowest activity was on February 5, 2019, the third day of the Lunar New Year, 
with the top five lowest values ranging from New Year's Eve to the fourth day of the **Lunar New Year**.
         """)

st.header('**Impact of Academic Calendar on Student Motivation and Retention in Exercise Programs: A Cohort Analysis**')
st.image('Heatmap.png')

st.write("""
Based on cohort analysis, retention rate of user doing exsercise increasing in September and February

Based on Taiwan academic calender, we can notice that:
the first day of school is 30 August (after summer holiday) winter holiday is 10 february - 16 february

We can conclude that student more motivated to learn on this platform after holiday and less motivated and getting bored few week after holiday.
         """)

st.header('**Analyzing Problem-Solving Patterns in Exercises**')
st.image('dificalty.png', caption='Distribution of Exercises by Difficulty')


# Load the data from the CSV file
df = pd.read_csv("exercises.csv")
st.image('User_count.png', caption='Distribution of problem attempts for students in this exercise')
st.write("""
We can observe a peak in the plots, most users do 5 or 6 problems in the exercise. This is mainly due to the proficiency mechanism, users would usually want to upgrade to level 1 and move onto the next exercise
""")

# Display the dataframe
st.write("## Distribution of hard Problems Done by Each Student", df)

# Select columns for x and y axes
x_axis = st.selectbox("Select the X-axis:", df.columns)
y_axis = st.selectbox("Select the Y-axis:", df.columns)

# Create the interactive bar plot
fig = px.bar(df, x=x_axis, y=y_axis, title=f"{y_axis} vs {x_axis}", labels={x_axis: x_axis, y_axis: y_axis})

# Display the plot in Streamlit
st.plotly_chart(fig)


st.header('**Analyzing Active Duration on junyi academy**')
st.image('10-Day.png', caption='Distribution of Login Periods in 10-Day Intervals')
st.write("""
the histogram shows how long users tend to stay engaged with the site and how long users remain active on a website.

as we observe in diagram above 30000 students activated in 10 days.that shows around 40% of students did not tend to continue this platform.
""")

st.header('**Investigating the Relationship Between Problem Difficulty and Time Spent**')
st.image('mean.png', caption='Mean Time for a User to Finish a Problem by Difficulty')
st.write("""
as we have the difficulty of problems in info_content dataset and the time spent in log_problem dataset,we merge these two to have some information about how much time does each problem take according to its difficulty.
we calculate the total time spent divided by the total number of problems attempted, it shows mean of time for user to finish the problem.
as green line shows,More students need more time to solve hard problems but in easy problems the line grows smoothly. if we compare the tree lines , we can observed 60000 students need up to 50 seconds to solve easy problems but in normal problems around 30000 students need this time.in hard problems about 20000 students need up to 50 seconds to solve questions
""")

st.header('**The Relationship Between Problem Difficulty and Correct Answer Rates**')
st.image('rate.png', caption='Distribution of Correct Rate by Difficulty')
st.markdown("""
<style>
.center-table {
  margin-left: auto;
  margin-right: auto;
  border-collapse: collapse;
  width: 50%;
}
.center-table th, .center-table td {
  border: 1px solid #ddd;
  padding: 8px;
  text-align: center;
}
.center-table th {
  text-align: center;
}
</style>
<table class="center-table">
  <tr>
    <th></th>
    <th>False</th>
    <th>True</th>
    <th>correct_rate</th>
  </tr>
  <tr>
    <td>Easy</td>
    <td>2943941</td>
    <td>8185414</td>
    <td>0.735480</td>
  </tr>
  <tr>
    <td>Normal</td>
    <td>1169029</td>
    <td>2014497</td>
    <td>0.632788</td>
  </tr>
  <tr>
    <td>Hard</td>
    <td>543970</td>
    <td>872778</td>
    <td>0.616043</td>
  </tr>
</table>
""", unsafe_allow_html=True)


st.write("""
We calculate the sum of correct answers and the total number of attempted.so we can measure the correct rate by dividing these two. The lines on the graph indicate that students generally have higher correct rates on easy problems. The distribution shows that as the difficulty increases, the correct rate decreases for most students. For easy problems, the curve is smoother and gradually increases, indicating less variability in correct rates among students. For hard problems, the curve is steeper in places, showing more variability and indicating that some students perform much better than others
         """)

st.header('**Temporal Patterns of User Engagement on junyi academy: Analyzing Hourly, Weekly, and Monthly Activity Trends**')
st.image('Hour.png', caption='Distribution of Hour')
st.image('week.png', caption='Distribution of Day of the Week')
st.image('month.png', caption='Distribution of Month')


# Title of the app
st.header('Implementation of Models', divider='rainbow')
st.write("""
#### Two time series Models :

1. "LSTM" (Long Short-Term Memory)

2. "SARIMA" (Seasonal Autoregressive Integrated Moving Average)

**Target variable:** The number of login in the junyi academy website
         """)

# Load the CSV file
df1 = pd.read_csv("login_count_D.csv")

# Display the DataFrame
st.dataframe(df1)
st.markdown("_Counts the amount of logins per day_")

# Ensure the date column is in datetime format
df1['date'] = pd.to_datetime(df1['date'])

# Create an interactive plot using Plotly
fig = px.line(df1, x='date', y='login_count', title='Logins per Day')

# Display the plot
st.plotly_chart(fig)

st.subheader("Log Transformations and the Augmented Dickey-Fuller (ADF) Test Stationarity")

# Display the descriptive text
st.markdown("""

The results of the ADF test are as follows:

- **Test Statistic**: -1.6759213947969376
- **p-value**: 0.44355923860017904
- **Number of Lags Used**: 7
- **Number of Observations**: 358

Higher p-value suggests weaker evidence against stationarity
""")
st.subheader("Assessing Time Series Stationarity Using Differencing and the Augmented Dickey-Fuller (ADF)")


# Display the descriptive text
st.markdown("""
The first difference of the login_count column was calculated to address its non-stationarity. To verify the effectiveness of this transformation, the Augmented Dickey-Fuller (ADF) test was applied to the differenced series.
The results of the ADF test are as follows:

- **Test Statistic**: -5.189843811972838
- **p-value**: 9.221691961925547e-06
- **Number of Lags Used**: 13
- **Number of Observations**: 351

confidently reject the null hypothesis and conclude that the differenced series is stationary""")

st.image('Stationary.png')

st.subheader("Achieving Stationarity through Log Differencing and ADF Testing in Time Series Analysis")

# Display the descriptive text
st.markdown("""
To confirm that the log-differenced series is stationary, we applied the Augmented Dickey-Fuller (ADF) test. The test yielded the following results:

- **p-value**: 0.0005584747014459555
""")

st.header('Visualizing Stationarity with Log-Transformed First Differences and ADF Testing')
st.image('Visualizing.png')


st.subheader('ACF and PACF')
# Definition section
st.write("""
Autocorrelation (also known as serial correlation) quantifies the linear relationship between a time series and its past observations""")
st.subheader("ACF")
st.write("""
This measures the correlation between a time series and lagged versions of itself. For example, ACF at lag 2 would compare the series with its own values shifted by two time points. It includes both the direct and indirect effects of past data points""")
st.image('acf.png')

# Interpretation section
st.subheader("PACF")
st.write("""
This measures the correlation between a time series and lagged versions of itself but after eliminating the variations already explained by the intervening comparisons. Essentially, PACF at lag 2 would measure the correlation between the series and its lagged version two time points back, but removing the effects of lags 1
""")
st.image('pacf.png')


st.subheader("Train and Test")

st.markdown("""

The oApproximately 70% of the data is used for training (train), and the remaining 30% is reserved for testing (test):

- **X_train shape: (254, 1)**
- **X_test shape: (109, 1)**
- **y_train shape: (254,)**
- **y_test shape: (109,)**
""")

st.image('ssarima.png')
st.image('sarimax.png')
st.write("""
The code uses the pm.auto_arima() function from the pmdarima library to automatically select an optimal ARIMA model for a given time series, y_train. This process involves trying different combinations of AR (Auto-Regressive) and MA (Moving Average) orders to find the best-fitting model based on the data.

The optimal model found is ARIMA(0,0,0), which essentially means that the model is a constant (with an intercept) and does not include any AR or MA terms. This implies that the time series does not show significant autocorrelation structure that AR or MA terms would capture.""")

st.subheader("Hyperparameter Tuning for SARIMAX Model")
st.image('hyper.png')
st.write("""
The goal of this code is to find the best combination of parameters for a Seasonal Autoregressive Integrated Moving Average (SARIMA) model using a brute-force approach.
It iterates through various combinations of SARIMA hyperparameters and evaluates their performance.
Parameter Combinations:
The code loops through different values of the following parameters:
p, i, and q: The non-seasonal AR, differencing, and MA orders.
P, D, and Q: The seasonal AR, seasonal differencing, and seasonal MA orders.
The seasonal period is set to 7 days (seasonal_order=(P, D, Q, 7)).
Model Fitting and Scoring:
For each parameter combination, it initializes and fits a SARIMAX model to the training data (train).
The model is configured with the specified orders and other settings (e.g., enforce_stationarity, enforce_invertibility).
It then forecasts the test data (test) and computes the R-squared score to evaluate model performance.""")

st.subheader("Visualizing SARIMA Model Forecasting and Evaluation")
st.image('sarima.png', caption='MAE: 0.17576978536911564 / MSE: 0.05663274146675312 / RMSE: 0.2379763464438286')
st.image('sarimaxx.png')
st.write("""
Statistical Summary:
The model_fit.summary() provides detailed information about the SARIMA model’s coefficients, standard errors, p-values, and other statistical metrics.
It’s useful for understanding the significance of each parameter.""")

st.subheader("Comprehensive Evaluation of SARIMA Model Forecasting with Visual and Statistical Diagnostics")
st.image('actual.png')
st.image('standard.png')
st.image('normal.png')

st.markdown("# LSTM MODEL")
# Title of the app

# LSTM Model Architecture section
st.subheader("LSTM Model Architecture")
st.write("""
- The LSTM model architecture is defined using the Sequential class from Keras.
- It consists of two LSTM layers with 128 and 64 units, respectively.
- The final output layer has a single unit.
""")

# Model Compilation section
st.subheader("Model Compilation")
st.write("""
- The model is compiled with the Adam optimizer and the mean squared error (MSE) loss function.
""")

# Model Training section
st.subheader("Model Training")
st.write("""
- The model is trained on the training data (train_X, train_Y) with a batch size of 3 and 20 epochs.
- Validation data (test_X, test_Y) is used for monitoring performance during training.
""")

# Prediction and Inversion section
st.subheader("Prediction and Inversion")
st.write("""
- The model predicts values for both training and test data.
- Predictions are inverted using the separate target scaler to obtain original scale values.
""")

# Evaluation Metrics section
st.subheader("Evaluation Metrics")
st.write("""
- The root mean squared error (RMSE) and mean absolute error (MAE) are calculated to assess model performance.
""")
st.image('lstmex.png')
# Title of the app
st.title("Time Series Forecast Visualization")

# Forecast Generation and Visualization section
st.subheader("Forecast Generation and Visualization")
st.write("""
The code generates a forecast for a time series model and visualizes the results along with the actual values. Here's a breakdown of the output you can expect:
""")
st.markdown("""

The root mean squared error (RMSE) and mean absolute error (MAE) are calculated to assess model performance:

- **RMSE: 0.3587078733959657**
- **MAE: 0.2896473026490036**
- **MSE: 0.12867133843625617**
""")

st.write("""
the LSTM model architecture is defined using the Sequential class from Keras. It consists of two LSTM layers with 128 and 64 units, respectively. The final output layer has a single unit. Model Compilation: The model is compiled with the Adam optimizer and the mean squared error (MSE) loss function. Model Training: The model is trained on the training data (train_X, train_Y) with a batch size of 3 and 20 epochs. Validation data (test_X, test_Y) is used for monitoring performance during training. Prediction and Inversion: The model predicts values for both training and test data. Predictions are inverted using the separate target scaler to obtain original scale values. Evaluation Metrics: The root mean squared error (RMSE) and mean absolute error (MAE) are calculated to assess model performance.
""")

# Actual vs Predicted Values section

# Combined Plot section
st.subheader("Combined Plot")
st.write("""
A single plot will be generated showing all the data and predictions together.
It will include lines for:
- Actual and predicted values for the training set
- Actual and predicted values for the test set
- The 12-period forecast (a dashed line) - This section is commented out, so you won't see it by default.
""")
st.image('combain.png')

st.subheader("Evaluating Training Performance: Actual vs Predicted Values in Time Series Forecasting")

st.write("""
This plot illustrates the comparison between the actual values of the training data and the values predicted by the time series model.
The x-axis represents the time, while the y-axis shows the values of the time series.
""")

st.image('predict.png')

st.title("Model Performance Comparison")

st.header("Performance Metrics")

st.subheader("SARIMA Model:")
st.write("**Mean Absolute Error (MAE):** 0.1621")
st.write("**Mean Squared Error (MSE):** 0.0503")
st.write("**Root Mean Squared Error (RMSE):** 0.2243")

st.subheader("LSTM Model:")
st.write("**Mean Absolute Error (MAE):** 0.2659")
st.write("**Mean Squared Error (MSE):** 0.1119")
st.write("**Root Mean Squared Error (RMSE):** 0.3346")

st.header("Analysis")

st.subheader("Mean Absolute Error (MAE):")
st.write("""
The SARIMA model has a lower MAE (0.1621) compared to the LSTM model (0.2659). 
This indicates that on average, the SARIMA model's predictions are closer to the actual values than those of the LSTM model.
""")

st.subheader("Mean Squared Error (MSE):")
st.write("""
The MSE for the SARIMA model (0.0503) is significantly lower than that for the LSTM model (0.1119). 
Since MSE penalizes larger errors more heavily, this suggests that the SARIMA model makes fewer large errors compared to the LSTM model.
""")

st.subheader("Root Mean Squared Error (RMSE):")
st.write("""
The RMSE of the SARIMA model (0.2243) is also lower than that of the LSTM model (0.3346). 
This metric, being in the same units as the data, provides a direct comparison of the error magnitude. 
Again, the SARIMA model demonstrates better performance.
""")

st.header("Conclusion")
st.write("""
Based on the evaluation metrics, the SARIMA model outperforms the LSTM model in this particular forecasting task. 
The SARIMA model exhibits lower MAE, MSE, and RMSE values, indicating that it provides more accurate and reliable predictions.
""")

st.write("""
The superior performance of the SARIMA model can be attributed to the inherent characteristics of the time series data. 
SARIMA is particularly effective in capturing seasonality and trends, which might be the dominant features in the dataset.
""")

st.write("""
On the other hand, while LSTM models are powerful and capable of capturing complex patterns and long-term dependencies in the data, 
they might require more tuning and larger datasets to outperform traditional methods like SARIMA in certain scenarios.
""")


