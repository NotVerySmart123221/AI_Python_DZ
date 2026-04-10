low_edu = df[df["education_num"] < 12]
high_edu = df[df["education_num"] >= 12]

low_mean = low_edu["hours_per_week"].mean()
high_mean = high_edu["hours_per_week"].mean()

plt.figure(1)
plt.bar(["low education", "high education"], [low_mean, high_mean])


# 2 


male = df[df["sex"] == "Male"]
female = df[df["sex"] == "Female"]

male_hours = male["hours_per_week"].mean()
female_hours = female["hours_per_week"].mean()

plt.figure(2)
plt.bar(["Male", "Female"], [male_hours, female_hours])


# 3


low_income = df[df["income"] == "<=50K"]
high_income = df[df["income"] == ">50K"]

low_gain = low_income["capital_gain"].mean()
high_gain = high_income["capital_gain"].mean()

plt.figure(3)
plt.bar(["<=50K", ">50K"], [low_gain, high_gain])

plt.show()
