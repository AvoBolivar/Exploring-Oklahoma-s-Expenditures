import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

# Press the green button in the gutter to run the script.
def line_of_best_fit(x, y):
    # this makes the 4 lists needed
    x_y = []
    x_squared = []
    for i in range(len(x)):
        x_y.append(x[i] * y[i])
        x_squared.append(pow(x[i], 2))

    # this adds up all the numbers in each list
    sum_of_x = sum(x)
    sum_of_y = sum(y)
    sum_x_y = sum(x_y)
    sum_x_squared = sum(x_squared)
    n = len(x)

    # the equation to find m
    slope = ((n * sum_x_y) - (sum_of_x * sum_of_y)) / (n * sum_x_squared - pow(sum_x_squared, 2))

    # the equation to find y intercept
    y_intercept = (sum_of_y - slope * sum_of_x) / n
    return slope, y_intercept


if __name__ == '__main__':

    # creates quick csv file
    table_one = pd.DataFrame({"Yes": [1, 2, 3], "No": [4, 5, 6]}, index=["Product 1", "Product 2", "Product 3"])

    # reads csv file
    f = pd.read_csv("expenditures-ledger-fy21.csv")
    files = ["data-expenditure-summary-fiscal-year-2015.csv",
             "data-expenditure-summary-fiscal-year-2016.csv",
             "data-expenditure-summary-fiscal-year-2017.csv",
             "data-expenditure-summary-fiscal-year-2018.csv",
             "data-expenditure-summary-fiscal-year-2019.csv",
             "expenditures2020.csv"]
    table_two = pd.read_csv(files[0])
    table_three = pd.read_csv(files[1])
    table_four = pd.read_csv(files[2])
    table_five = pd.read_csv(files[3])
    table_six = pd.read_csv(files[4])
    table_seven = pd.read_csv(files[5])

    data = [table_two, table_three, table_four, table_five, table_six, table_seven]

    # gets the name of the columns
    header = list(f.columns.unique())
    unique_orgs = []
    occurrence_of_fund = {}
    percentage_of_fund = {}
    money_spent_per_fund = {}
    # this gets every agency name
    for i in f[header[1]]:
        if i not in unique_orgs:
            unique_orgs.append(i)

    # for i in range(len(unique_orgs)):
        # print(str(i)+": "+unique_orgs[i])

    # this gets type of fund and assigns a value to it
    for j in f[header[-1]]:
        if j in occurrence_of_fund:
            occurrence_of_fund[j] += 1
        else:
            occurrence_of_fund[j] = 1

    # this gets the percentage of amount of times spent
    total = 0
    for i in occurrence_of_fund:
        total += occurrence_of_fund[i]
    for j in occurrence_of_fund:
        percentage_of_fund[j] = round(occurrence_of_fund[j]/total*100, 2)

    # for i in percentage_of_fund:
    #    print(i+" : "+str(percentage_of_fund[i])+"% -- "+str(occurrence_of_fund[i]))

    # this gets the amount of total money spent, most spent, and least spent
    total_money = 0
    max_amount = 0
    min_amount = 100000000

    for i in range(len(f[header[-3]])):
        total += f[header[-3]][i]
        if f[header[-3]][i] > max_amount:
            max_amount = f[header[-3]][i]
        elif f[header[-3]][i] < min_amount:
            min_amount = f[header[-3]][i]
        if f[header[-1]][i] in money_spent_per_fund:
            money_spent_per_fund[f[header[-1]][i]] += f[header[-3]][i]
        else:
            money_spent_per_fund[f[header[-1]][i]] = f[header[-3]][i]

    # print(f"max: {max_amount} \nmin: {min_amount}")
    # average amount each type of fund usually gave out
    # for j in money_spent_per_fund:
        # print(j+" : "+str(round(money_spent_per_fund[j]/occurrence_of_fund[j], 2))+"$")
    # this gets the amount the gov spent on normal education
    index_of_education = []
    education_expenses = 0
    for i in range(len(f[header[1]])):
        if f[header[1]][i] == unique_orgs[28]:
            index_of_education.append(i)

    for j in index_of_education:
        education_expenses += f[header[-3]][j]

    # print(education_expenses)

    # index for colleges in unique orgs
    # 12, 45, 58, 67, 69, 86, 90, 103, 119, 121, 120, 122, 123, 124, 125, 145 - 154,
    index_higher_education = [12, 45, 58, 67, 69, 86, 90, 103, 119, 121, 120, 122, 123, 124, 125, 145, 146, 147,
                              148, 149, 150, 151, 152, 153, 154]
    higher_education_clubs = []
    # this puts the name of clubs all in one list
    for i in index_higher_education:
        higher_education_clubs.append(unique_orgs[i])

    higher_education_with_prices = {}
    for i in range(len(f[header[1]])):
        if f[header[1]][i] in higher_education_clubs:
            if f[header[1]][i] in higher_education_with_prices:
                higher_education_with_prices[f[header[1]][i]] += f[header[-3]][i]
            else:
                higher_education_with_prices[f[header[1]][i]] = f[header[-3]][i]

    values = higher_education_with_prices.values()
    spent_higher_education = sum(values)

    # for i in higher_education_with_prices:
        # print(i+": "+str(higher_education_with_prices[i]))
    # print("Money spent on higher education: "+str(spent_higher_education))
    # print("Money spent on public k - 12 schools: "+str(education_expenses))
    # difference = spent_higher_education- education_expenses
    # print(f"\n{difference} is how much more was spent on colleges")

    # 698,696 amount of kids in public schools in Oklahoma
    # 214, 200 kids in college

    # trying to predict future wild life expenses with linear regression
    # looking at 3 different agencies
    # index
    # 40: DEPARTMENT OF WILDLIFE CONSERVATION
    # 41: DEPT OF AGRICULTURE FOOD & FORESTRY
    # 42: DEPT. OF ENVIRONMENTAL QUALITY

    # x values of each will be the months since January 2021
    # y values of each will be the amount of money spent that month
    dp_wild_life = []
    WildLife_cost_per_month = {}
    WL_2015 = {}
    WL_2016 = {}
    WL_2017 = {}
    WL_2018 = {}
    WL_2019 = {}
    WL_2020 = {}

    di_of_data = [WL_2015, WL_2016, WL_2017, WL_2018, WL_2019, WL_2020]

    indexes = [unique_orgs[40], unique_orgs[41], unique_orgs[42]]

    for i in range(len(f[header[1]])):
        if f[header[1]][i] == unique_orgs[40]:
            if f[header[-4]][i] in WildLife_cost_per_month:
                WildLife_cost_per_month[f[header[-4]][i]] += f[header[-3]][i]
            else:
                WildLife_cost_per_month[f[header[-4]][i]] = f[header[-3]][i]

    # for i in WildLife_cost_per_month:
        # print(str(i)+" : "+str(WildLife_cost_per_month[i]))

    # i need how much the gov spent since on department of wildlife since 2016
    # data is the name of a list which contains the csv files
    for i in range(len(data)):
        # creates header for the particular data set
        head = list(data[i].columns.unique())
        index_of_agency = data[i][head[1]]
        index_of_month = data[i][head[-4]]
        index_of_price = data[i][head[-3]]
        for j in range(len(index_of_agency)):
            # looping through the datasets departments looking for wildlife one
            # adding the month and amount spent that month to a dictionary
            if index_of_agency[j] == unique_orgs[40]:
                if i == 0:
                    amount = float(index_of_price[j][1:])
                else:
                    amount = index_of_price[j]
                if index_of_month[j] in di_of_data[i]:
                    if amount >= 0:
                        di_of_data[i][data[i][head[-4]][j]] += amount
                else:
                    if amount >= 0:
                        di_of_data[i][data[i][head[-4]][j]] = amount

    count = 0
    # this is my y
    expenses_since_2015 = []
    # this is my x
    months_since_2015 = list(range(1, 73))
    for i in di_of_data:
        for j in i:
            expenses_since_2015.append(i[j])

    # bellow is trying to use sklearn which i failed and will have to try again
    x_train_variables = list(range(1, 9))
    y_variables = list(WildLife_cost_per_month.values())
    y_train_variables = y_variables[:8]
    matrix_X_train = np.array(list(zip(x_train_variables, y_train_variables)))

    x_pr_variables = list(range(9, 12))
    y_pr_variables = y_variables[9:12]
    matrix_X_pridiction = np.array(list(zip(x_pr_variables, y_pr_variables)))
    # setting up regression model
    reg = linear_model.LinearRegression()
    reg.fit(matrix_X_train, y_train_variables)
    y_pridiction_variables = reg.predict(matrix_X_pridiction)

    # The coefficients
    # print("Coefficients: \n", reg.coef_)

    # plt.scatter(x_train_variables, y_train_variables, color="black")
    # plt.plot(x_pr_variables, y_pridiction_variables, color="blue")

    # plt.xticks(())
    # plt.yticks(())

    # plt.show()

    x_key = list(range(73, 85))
    y_value = list(WildLife_cost_per_month.values())

    y_predicted = []
    older_y_predicted = []
    # this is getting line of best fit using a function i built
    slope, y_intercept = line_of_best_fit(x_key, y_value)
    older_slope, older_y_intercept = line_of_best_fit(months_since_2015, expenses_since_2015)
    older_y_intercept = round(older_y_intercept, 3)
    older_slope = round(older_slope)
    slope = round(slope, 3)
    y_intercept = round(y_intercept, 3)
    print(f"y= {slope}x + {y_intercept}")
    print(f"y= {older_slope}x + {older_y_intercept}")
    equation = lambda x: x*slope + y_intercept
    older_equation = lambda y: y*older_slope + older_y_intercept
    for i in x_key:
        y_predicted.append(equation(i))
    for i in months_since_2015:
        older_y_predicted.append(older_equation(i))

    # getting the ss(mean)
    temp = 0
    avg_of_exp_since_2015 = sum(expenses_since_2015)/len(expenses_since_2015)
    for i in expenses_since_2015:
        temp += pow((i-avg_of_exp_since_2015), 2)
        ss_mean = temp/len(expenses_since_2015)

    # getting the ss(fit)
    temp = 0
    for i in range(len(older_y_predicted)):
        temp += pow((older_y_predicted[i] - expenses_since_2015[i]), 2)
        ss_fit = temp/len(expenses_since_2015)

    # getting the variance
    variance = round(((ss_mean - ss_fit)/ss_mean)*100, 5)
    print("Variance: " + str(variance))

    print(x_train_variables, y_predicted)
    plt.scatter(x_key, y_value, color="black")
    plt.scatter(months_since_2015, expenses_since_2015, color="orange")
    plt.plot(y_predicted, color="blue")
    plt.plot(older_y_predicted, color="purple")

    plt.xticks(())
    plt.yticks(())

    plt.show()
