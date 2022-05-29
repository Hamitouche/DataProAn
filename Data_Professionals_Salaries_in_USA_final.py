###########################################################################################################
######## author = Thinhinane
######## layout inspired by https://share.streamlit.io/tylerjrichards/streamlit_goodreads_app/books.py
###########################################################################################################
import streamlit as st
import numpy as np
import pandas as pd
import pickle
import time
from matplotlib import pyplot as plt
from  matplotlib.ticker import FuncFormatter
import seaborn as sns
import pandas as pd
from datetime import date
import numpy as np
import math


#Import and clean first dataset
data=pd.read_csv("data_cleaned_2021.csv", index_col=0)
data=data[~(data["Salary Estimate"].str.contains("Per Hour"))]
data=data[['Job Title','Job Description', 'Rating','Location','Job Location','Headquarters', 'Size', 'Founded',
       'Type of ownership', 'Industry', 'Sector', 'Revenue','Lower Salary', 'Upper Salary',
       'Avg Salary(K)', 'company_txt','Age', 'Python',
       'spark', 'aws', 'excel', 'sql', 'sas', 'keras', 'pytorch', 'scikit',
       'tensor', 'hadoop', 'tableau', 'bi', 'flink', 'mongo', 'google_an']]
data['Age']=date.today().year-data['Founded']
data=data.rename(columns={'company_txt':'Company Name', 'Age':'Company Age', "google_an":"google analytics", "Location":"City", "Job Location": "State Code", "Avg Salary(K)":"Average Salary (K USD)", "Lower Salary":"Lower Salary (K USD)","Upper Salary":"Upper Salary (K USD)", "Rating":"Job Rating"})
data=data[data['Sector']!='-1']
data=data[data['Founded']!=-1]
data=data[data["Size"]!="unknown"]
data["Job Title"]=data["Job Title"].transform(lambda x: x.replace("Sr.", "Senior") if x.startswith("Sr. ") else x)
data["Job Title"]=data["Job Title"].transform(lambda x: x.replace("Sr", "Senior") if x.startswith("Sr ") else x)

### Data jobs######
df_jobs=data["Job Title"].value_counts(sort=True)
df_jobs=df_jobs.reset_index()
df_jobs=df_jobs.rename(columns={"index":"Job Title","Job Title":"Number"})


#### Data Average Salaries####
df_average_salaries=data.groupby("Job Title").agg({"Average Salary (K USD)": "mean"})
df_average_salaries=df_average_salaries.reset_index()
df_average_salaries= df_average_salaries.sort_values("Average Salary (K USD)", ascending=False)


### Data Max Salary ####
df_max_salaries=data.groupby("Job Title").agg({"Upper Salary (K USD)": "max"})
df_max_salaries=df_max_salaries.reset_index()
df_max_salaries= df_max_salaries.sort_values("Upper Salary (K USD)", ascending=False)


### Data Min Salary ####
df_min_salaries=data.groupby("Job Title").agg({"Lower Salary (K USD)": "max"})
df_min_salaries=df_min_salaries.reset_index()
df_min_salaries= df_min_salaries.sort_values("Lower Salary (K USD)", ascending=False)


### Data Jobs Salries###
df_jobs_average_salaries=df_jobs.merge(df_average_salaries, on="Job Title")
df_jobs_max_salaries=df_jobs.merge(df_max_salaries, on="Job Title")
df_jobs_min_salaries=df_jobs.merge(df_min_salaries, on="Job Title")


#Import and clean second dataset (states)
states=pd.read_csv("state-abbrevs.csv")
states=states.rename(columns={"state":"State Name", "abbreviation":"State Code"})

#Merge datasets
data=data.merge(states, how="left", on="State Code")
st.set_page_config(layout="wide")


####################
### INTRODUCTION ###
####################

row0_spacer1, row0_1, row0_spacer2, row0_2, row0_spacer3 = st.columns((.1, 2.3, .1, 1.3, .1))
with row0_1:
    st.title('DataProAn - Data Professionals Salaries Analyzer for US')
with row0_2:
    st.text("")
    st.subheader('Streamlit App by [Thinhinane](https://www.linkedin.com/in/thinhinane-hamitouche/)')
row3_spacer1, row3_1, row3_spacer2 = st.columns((.1, 3.2, .1))
with row3_1:
    st.markdown("Hey there! Have you ever thought about starting a career as a data expert  in the USA and had no idea about different     opportunities and salaries ? However, you could not start your research because you did not have any stats at hand ? Well, this interactive application containing Glassdoor data of 2021 allows you to discover all that ! ")
    st.markdown("You can find the source code in the [DataProAn GitHub Repository](https://github.com/Hamitouche/DataProAn)")
    

#################
### SELECTION ###
#################
### Helper Methods ###
def get_boundary_salary(df_data):
    min_salary = math.floor(np.min(data["Average Salary (K USD)"]))
    max_salary= math.ceil(np.max(data["Average Salary (K USD)"]))
    boundary_salary=list(range(min_salary, max_salary+1))
    return boundary_salary

def filtered_salary(df_data, born_inf, born_sup):
    condition=(df_data["Average Salary (K USD)"] <=born_sup) & (df_data["Average Salary (K USD)"] >=born_inf)
    df_data=df_data[condition]
    return df_data

st.sidebar.text('')
st.sidebar.text('')
st.sidebar.text('')
### SALARY RANGE ###
st.sidebar.markdown("**First select the data range you want to analyze:** 👇")
boundary_salary = get_boundary_salary(data)
start_salary, end_salary = st.sidebar.select_slider('Select the average salary (in K USD) range you want to include ', boundary_salary,value=[min(boundary_salary),max(boundary_salary)])

data=filtered_salary(data, start_salary,end_salary )        

      
    
def filtered_states(df_data):
    return df_data[df_data["State Name"].isin(selected_states)]


def get_unique_states(df_data):
    return np.unique(df_data["State Name"]).tolist()

### STATE SELECTION ###
unique_states = get_unique_states(data)
all_states_selected = st.sidebar.selectbox('Do you want to only include specific states ? If the answer is yes, please check the box below and then select the state(s) in the new field.', ['Include all available states','Select states manually (choose below)'])
if all_states_selected == 'Select states manually (choose below)':
    selected_states = st.sidebar.multiselect("Select and deselect the states you would like to include in the analysis. You can clear the current selection by clicking the corresponding x-button on the right", unique_states, default = unique_states)
    data = filtered_states(data)     


### SEE DATA ###
row6_spacer1, row6_1, row6_spacer2 = st.columns((.2, 7.1, .2))
with row6_1:
    st.subheader("Currently selected data:")

row2_spacer1, row2_1, row2_spacer2, row2_2, row2_spacer3, row2_3, row2_spacer4, row2_4, row2_spacer5   = st.columns((.2, 1.6, .2, 1.6, .2, 1.6, .2, 1.6, .2))
with row2_1:
    unique_rows_in_df = len(data.index)
    str_rows = "✅ " + str(unique_rows_in_df) + " Matches"
    st.markdown(str_rows )
with row2_2:
    unique_job_titles_in_df = len(np.unique(data["Job Title"]).tolist())
    str_job_titles = "🧑‍💼" + str(unique_job_titles_in_df) + " Job Titles"
    st.markdown(str_job_titles )
with row2_3:
    unique_states_in_df = len(np.unique(data["State Code"]).tolist())
    str_states = "🌎" + str(unique_states_in_df) + " States"
    st.markdown(str_states)
with row2_4:
    avg_salary_in_df = round(data["Average Salary (K USD)"].mean())
    str_salary = "💸 " + str(avg_salary_in_df) + " (K USD)"
    st.markdown(str_salary)

row3_spacer1, row3_1, row3_spacer2 = st.columns((.2, 7.1, .2))
with row3_1:
    st.markdown("")
    see_data = st.expander('You can click here to see the raw data first 👉')
    with see_data:
        st.dataframe(data=data)
st.text('')


################
### ANALYSIS ###
################
def get_unique_title_jobs(df_data):
    return np.unique(df_data["Job Title"]).tolist()
    


### DATA EXPLORER ###
unique_title_jobs=get_unique_title_jobs(data)
row12_spacer1, row12_1, row12_spacer2 = st.columns((.2, 7.1, .2))

with row12_1:
    st.subheader('Match Finder')
    st.markdown('Make comparaisons between job titles...')  

row13_spacer1, row13_1, row13_spacer2, row13_2, row13_spacer3, row13_3, row13_spacer4   = st.columns((.2, 2.3, .2, 2.3, .2, 2.3, .2))
with row13_2:
        first_job = st.selectbox ("", unique_title_jobs,key = 'hi_lo') 
with row13_3:
        second_job = st.selectbox ("", unique_title_jobs,key = 'what')
        
data_filtred_first_job=data[data["Job Title"]==first_job]
data_filtred_second_job=data[data["Job Title"]==second_job]
row16_spacer1, row16_1, row16_2, row16_3, row16_4,row16_5, row16_spacer2 = st.columns((0.5, 1.5, 1.5, 1, 2, 1, 0.5))
with row16_1:
        st.markdown("🧑‍💼 Number of Jobs")
        st.markdown("↘️ Lower Salary ( K USD)")
        st.markdown("↗️ Upper Salary (K USD)")
        st.markdown("💸 Average Salary (K USD)")
with row16_3:
        st.markdown(" ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎"+str(len(data_filtred_first_job.index)))
        st.markdown(" ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎"+str(np.min(data_filtred_first_job["Lower Salary (K USD)"])))
        st.markdown(" ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎‎"+str(np.max(data_filtred_first_job["Upper Salary (K USD)"])))
        st.markdown(" ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎‎ ‎‎"+str(round(np.mean(data_filtred_first_job["Average Salary (K USD)"]))))

with row16_5:
        st.markdown(" ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎"+str(len(data_filtred_second_job.index)))
        st.markdown(" ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎"+str(np.min(data_filtred_second_job["Lower Salary (K USD)"])))
        st.markdown(" ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎‎"+str(np.max(data_filtred_second_job["Upper Salary (K USD)"])))
        st.markdown(" ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎‎"+str(round(np.mean(data_filtred_second_job["Average Salary (K USD)"]))))



### Job Title ###

def plot_Number_Jobs_per_Job_Title(attr): #total #against, #conceived
    rc = {'figure.figsize':(8,4.5),
          'axes.facecolor':'#0e1117',
          'axes.edgecolor': '#0e1117',
          'axes.labelcolor': 'white',
          'figure.facecolor': '#0e1117',
          'patch.edgecolor': '#0e1117',
          'text.color': 'white',
          'xtick.color': 'white',
          'ytick.color': 'white',
          'grid.color': 'grey',
          'font.size' : 8,
          'axes.labelsize': 12,
          'xtick.labelsize': 8,
          'ytick.labelsize': 12}
    
    plt.rcParams.update(rc)
    fig, ax = plt.subplots()
    ### Retrieve data
    df_plot=data["Job Title"].value_counts(sort=True)
    df_plot=df_plot.reset_index()
    df_plot=df_plot.rename(columns={"index":"Job Title","Job Title":"Number"})
    df_plot=df_plot.iloc[0:attr,:]
    ax = sns.barplot(x="Job Title", y="Number", data=df_plot, color = "#b80606")
    ax.set(xlabel = "Job Title", ylabel ="Number of jobs" )
    plt.xticks(rotation=66,horizontalalignment="right")
    for p in ax.patches:
            ax.annotate(format(str(int(p.get_height()))), 
                  (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha = 'center',
                   va = 'center', 
                   xytext = (0, 18),
                   rotation = 90,
                   textcoords = 'offset points')
    st.pyplot(fig)

#### Writing results #########
row4_spacer1, row4_1, row4_spacer2 = st.columns((.2, 7.1, .2))
with row4_1:
    st.subheader('Analysis per Job Title')
row5_spacer1, row5_1, row5_spacer2, row5_2, row5_spacer3  = st.columns((.2, 2.3, .4, 4.4, .2))
with row5_1:
    st.markdown('Investigate opportunities for each offer. Which Job Title is the most common ? You can go from 1 to 10 most common job titles. ')    
    attribute_job_title = st.selectbox ("How many job titles do you want to analyze ?", [1,2,3,4,5,6,7,8,9,10], key = 'attribute_job_title')
with row5_2:
    plot_Number_Jobs_per_Job_Title(attribute_job_title)


#### SALARY ###

def plot_salary_per_job_title(attr,measure):
    rc = {'figure.figsize':(8,4.5),
          'axes.facecolor':'#0e1117',
          'axes.edgecolor': '#0e1117',
          'axes.labelcolor': 'white',
          'figure.facecolor': '#0e1117',
          'patch.edgecolor': '#0e1117',
          'text.color': 'white',
          'xtick.color': 'white',
          'ytick.color': 'white',
          'grid.color': 'grey',
          'font.size' : 8,
          'axes.labelsize': 12,
          'xtick.labelsize': 8,
          'ytick.labelsize': 12}
    plt.rcParams.update(rc)
    fig, ax = plt.subplots()
     ### Retrieve data
   
    if measure == "Mean":
        df_plot=data.groupby("Job Title").agg({"Average Salary (K USD)": "mean"})
        df_plot=df_plot.reset_index()
        df_plot= df_plot.sort_values("Average Salary (K USD)", ascending=False)
        df_plot=df_plot.iloc[0:attr,:]
        y_str = "Average Salary (K USD)"
        ax = sns.barplot(x="Job Title", y="Average Salary (K USD)", data=df_plot, color = "#b80606")
    if measure == "Maximum":
        df_plot=data.groupby("Job Title").agg({"Upper Salary (K USD)": "max"})
        df_plot=df_plot.reset_index()
        df_plot= df_plot.sort_values("Upper Salary (K USD)", ascending=False)
        df_plot=df_plot.iloc[0:attr,:]
        y_str = "Upper Salary (K USD)"
        ax = sns.barplot(x="Job Title", y="Upper Salary (K USD)", data=df_plot, color = "#b80606")
    if measure=="Minimum" :
        df_plot=data.groupby("Job Title").agg({"Lower Salary (K USD)": "min"})
        df_plot=df_plot.reset_index()
        df_plot=df_plot.iloc[0:attr,:]
        df_plot= df_plot.sort_values("Lower Salary (K USD)", ascending=True)
        y_str ="Lower Salary (K USD)"
        ax = sns.barplot(x="Job Title", y="Lower Salary (K USD)", data=df_plot, color = "#b80606")
    ax.set(xlabel = "Job Title", ylabel = y_str)
   
    plt.xticks(rotation=66,horizontalalignment="right")
    for p in ax.patches:
            ax.annotate(format(str(int(p.get_height()))), 
                  (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha = 'center',
                   va = 'center', 
                   xytext = (0, 18),
                   rotation = 90,
                   textcoords = 'offset points')
    st.pyplot(fig)
        

#### Writing results #########
row6_spacer1, row6_1, row6_spacer2 = st.columns((.2, 7.1, .2))
with row6_1:
    st.subheader('Analysis of Salaries per Job Title')
row7_spacer1, row7_1, row7_spacer2, row7_2, row7_spacer3  = st.columns((.2, 2.3, .4, 4.4, .2))
with row7_1:
    st.markdown('Investigate salaries. Which job is the most paid ? Show upper, lower and average salaries. ')    
    attribute_salary=st.selectbox ("How many job titles do you want to analyze ?", [1,2,3,4,5,6,7,8,9,10], key = 'attribute_salary')
    salary_measure = st.selectbox ("Which measure do you want to analyze ?", ["Mean", "Maximum", "Minimum"], key="salary_measure")
with row7_2:
    plot_salary_per_job_title(attribute_salary,salary_measure)

### Location, Sector, industry ###
def plot_measure_per_category(attr,measure):
    rc = {'figure.figsize':(8,4.5),
          'axes.facecolor':'#0e1117',
          'axes.edgecolor': '#0e1117',
          'axes.labelcolor': 'white',
          'figure.facecolor': '#0e1117',
          'patch.edgecolor': '#0e1117',
          'text.color': 'white',
          'xtick.color': 'white',
          'ytick.color': 'white',
          'grid.color': 'grey',
          'font.size' : 8,
          'axes.labelsize': 12,
          'xtick.labelsize': 8,
          'ytick.labelsize': 12}
    plt.rcParams.update(rc)
    fig, ax = plt.subplots()
     ### Retrieve data
    if attr=="State" :
        attr="State Name"
    if measure == "Number of jobs":
        df_plot=data[attr].value_counts(sort=True)
        df_plot=df_plot.reset_index()
        df_plot=df_plot.rename(columns={"index":attr,attr:"Number"})
        df_plot=df_plot.iloc[0:10,:]
        ax = sns.barplot(x=attr, y="Number", data=df_plot, color = "#b80606")
        ax.set(xlabel = attr, ylabel ="Number of jobs" )
        
    if measure == "Average Salary":
        df_plot=data.groupby(attr).agg({"Average Salary (K USD)": "mean"})
        df_plot=df_plot.reset_index()
        df_plot= df_plot.sort_values("Average Salary (K USD)", ascending=False)
        df_plot=df_plot.iloc[0:10,:]
        y_str = "Average Salary (K USD)"
        ax = sns.barplot(x=attr, y="Average Salary (K USD)", data=df_plot, color = "#b80606")
        ax.set(xlabel = attr, ylabel ="Average Salary (K USD)" )
    plt.xticks(rotation=66,horizontalalignment="right")
    for p in ax.patches:
            ax.annotate(format(str(int(p.get_height()))), 
                  (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha = 'center',
                   va = 'center', 
                   xytext = (0, 18),
                   rotation = 90,
                   textcoords = 'offset points')
    st.pyplot(fig)


#### Writing results #########
row8_spacer1, row8_1, row8_spacer2 = st.columns((.2, 7.1, .2))
with row8_1:
    st.subheader('Analysis per category')
row9_spacer1, row9_1, row9_spacer2, row9_2, row9_spacer3  = st.columns((.2, 2.3, .4, 4.4, .2))
with row9_1:
    st.markdown('Investigate stats over multiple categorical variables. How opportunities and salaries are related to them ? ')    
    category = st.selectbox ("Which aspect do you want to analyze ?", ["City","State","Company Name", "Industry","Sector","Type of ownership"], key = 'category')
    category_measure = st.selectbox ("Which measure do you want to analyze?", ["Number of jobs", "Average Salary"], key = 'category_measure')
with row9_2:
    plot_measure_per_category(category, category_measure)
    
### CORRELATION ###
label_attr_dict_correlation={'Average Salary':'Average Salary (K USD)','Upper Salary':'Upper Salary (K USD)', 'Lower Salary':'Lower Salary (K USD)','Company Age':'Company Age', 'Company Name':'Company Name', 'Job Rating':'Job Rating', 'Number of jobs':'Number'}
def plt_attribute_correlation(aspect1, aspect2):
    df_plot=data
    rc = {'figure.figsize':(5,5),
          'axes.facecolor':'#0e1117',
          'axes.edgecolor': '#0e1117',
          'axes.labelcolor': 'white',
          'figure.facecolor': '#0e1117',
          'patch.edgecolor': '#0e1117',
          'text.color': 'white',
          'xtick.color': 'white',
          'ytick.color': 'white',
          'grid.color': 'grey',
          'font.size' : 8,
          'axes.labelsize': 12,
          'xtick.labelsize': 12,
          'ytick.labelsize': 12}
    plt.rcParams.update(rc)
    fig, ax = plt.subplots()
    asp1 = label_attr_dict_correlation[aspect1]
    asp2 = label_attr_dict_correlation[aspect2]
    if ((corr_type=="Regression Plot (Recommended)") & (aspect2=="Average Salary") & (aspect1=="Number of jobs")):
        ax = sns.regplot(x=asp1, y=asp2, x_jitter=.1, data=df_jobs_average_salaries, color = '#f21111',scatter_kws={"color": "#f21111"},line_kws={"color": "#c2dbfc"})
    elif ((corr_type=="Regression Plot (Recommended)") & (aspect2=="Lower Salary") & (aspect1=="Number of jobs")):
        ax = sns.regplot(x=asp1, y=asp2, x_jitter=.1, data=df_jobs_min_salaries, color = '#f21111',scatter_kws={"color": "#f21111"},line_kws={"color": "#c2dbfc"})
    elif ((corr_type=="Regression Plot (Recommended)") & (aspect2=="Upper Salary") & (aspect1=="Number of jobs")):
        ax = sns.regplot(x=asp1, y=asp2, x_jitter=.1, data=df_jobs_max_salaries, color = '#f21111',scatter_kws={"color": "#f21111"},line_kws={"color": "#c2dbfc"})
    elif(corr_type=="Regression Plot (Recommended)"):
        ax = sns.regplot(x=asp1, y=asp2, x_jitter=.1, data=df_plot, color = '#f21111',scatter_kws={"color": "#f21111"},line_kws={"color": "#c2dbfc"})
    elif((corr_type=="Standard Scatter Plot")& (aspect2=="Average Salary") & (aspect1=="Number of jobs") ):
        ax = sns.scatterplot(x=asp1, y=asp2, data=df_jobs_average_salaries, color = '#f21111')
    elif((corr_type=="Standard Scatter Plot")& (aspect2=="Upper Salary") & (aspect1=="Number of jobs") ):
        ax = sns.scatterplot(x=asp1, y=asp2, data=df_jobs_max_salaries, color = '#f21111')
    elif((corr_type=="Standard Scatter Plot")& (aspect2=="Lower Salary") & (aspect1=="Number of jobs") ):
        ax = sns.scatterplot(x=asp1, y=asp2, data=df_jobs_min_salaries, color = '#f21111')
    elif(corr_type=="Standard Scatter Plot"):
        ax = sns.scatterplot(x=asp1, y=asp2, data=df_plot, color = '#f21111', hue='Size')
    ax.set(xlabel = aspect1, ylabel = aspect2)
    st.pyplot(fig, ax)

### Writing results ###
corr_plot_types = ["Regression Plot (Recommended)","Standard Scatter Plot"] 

row10_spacer1, row10_1, row10_spacer2 = st.columns((.2, 7.1, .2))
with row10_1:
    st.subheader('Correlation Analysis')
row11_spacer1, row11_1, row11_spacer2, row11_2, row11_spacer3  = st.columns((.2, 2.3, .4, 4.4, .2))
with row11_1:
    st.markdown('Investigate the correlation of attributes, but keep in mind correlation does not imply causation. Is there any correlation between company age and job rating on one hand and opportunities and average salary in the other hand ? ')    
    corr_type = st.selectbox ("What type of correlation plot do you want to see ?", corr_plot_types)
    y_axis_aspect2 = st.selectbox ("Which attribute do you want on the y-axis ?", ["Average Salary", "Lower Salary", "Upper Salary"])
    x_axis_aspect1 = st.selectbox ("Which attribute do you want on the x-axis ?", ["Company Age", "Job Rating", "Number of jobs"])
with row11_2:
    plt_attribute_correlation(x_axis_aspect1, y_axis_aspect2)
    
    
## Skills #####
def plot_score_per_skill():
    rc = {'figure.figsize':(8,4.5),
          'axes.facecolor':'#0e1117',
          'axes.edgecolor': '#0e1117',
          'axes.labelcolor': 'white',
          'figure.facecolor': '#0e1117',
          'patch.edgecolor': '#0e1117',
          'text.color': 'white',
          'xtick.color': 'white',
          'ytick.color': 'white',
          'grid.color': 'grey',
          'font.size' : 8,
          'axes.labelsize': 12,
          'xtick.labelsize': 8,
          'ytick.labelsize': 12}
    
    plt.rcParams.update(rc)
    fig, ax = plt.subplots()
    ### Retrieve data
    df_plot=data.loc[:,"Python":"google analytics"].agg(sum)
    df_plot=df_plot.to_frame().reset_index()
    df_plot=df_plot.rename(columns={"index":"Skill", 0:"Score"})
    df_plot=df_plot.sort_values("Score", ascending=False)
    ax = sns.barplot(x="Score", y="Skill", data=df_plot, color = "#b80606")
    ax.set(xlabel = "Score", ylabel ="Skill & Technologie" )
    for p in ax.patches:
            ax.annotate(format(str(int(p.get_height()))), 
                  (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha = 'center',
                   va = 'center', 
                   xytext = (0, 18),
                   rotation = 90,
                   textcoords = 'offset points')
    st.pyplot(fig)

### Writing results ###
corr_plot_types = ["Regression Plot (Recommended)","Standard Scatter Plot"] 

row11_spacer1, row11_1, row11_spacer2 = st.columns((.2, 7.1, .2))
with row11_1:
    st.subheader('Skills Analysis')
row12_spacer1, row12_1, row12_spacer2, row12_2, row12_spacer3  = st.columns((.2, 2.3, .4, 4.4, .2))
with row12_1:
    st.markdown('Investigate required skills and technologies, but keep in mind that technology evolves. Which score can we give to them ?')    
with row12_2:
    plot_score_per_skill()