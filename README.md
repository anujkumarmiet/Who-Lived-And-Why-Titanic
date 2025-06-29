# Who Lived and Why : An Analytical Breakdown of Titanic Passengers
A hands-on exploratory data analysis (EDA) project on the iconic Titanic dataset. This notebook dives deep into understanding survival patterns by uncovering trends, engineering features, and crafting a data-driven narrative.

---

## 📘 About the Dataset

The dataset contains information on passengers aboard the Titanic, including demographics, class, fare, and survival status. It's a classic for classification and feature engineering challenges.

**Key Columns:**
- `Survived`: Binary outcome (0 = No, 1 = Yes)
- `Pclass`, `Sex`, `Age`, `Fare`: Core demographic and socio-economic attributes
- `SibSp`, `Parch`: Family relationships
- `Ticket`, `Cabin`, `Embarked`: Semi-structured/messy fields

---

## 📊 Exploratory Data Analysis (EDA)

### 1. 🔹 Data Overview & Cleaning
```python
df = pd.read_csv('train.csv')
df.head()
df.info()
df.isnull().sum()
```

### 2. 🔹 Univariate Analysis (Numerical)
```python
# Age Distribution
df['Age'].plot(kind='hist', bins=20)
df['Age'].plot(kind='kde')
df['Age'].plot(kind='box')
print(df['Age'].describe())
```

📌 **Findings:**
- Nearly **20% Age data missing**
- Distribution is approximately **normal** with slight right skew
- Outliers detected (> 65 years)

### 3. 🔹 Feature Engineering: `individual_fare`
```python
df['individual_fare'] = df['Fare'] / (df['SibSp'] + df['Parch'] + 1)
```

📌 **Motivation**: Fare reflects **group fare**, not per-person—normalizing improves downstream modeling.

### 4. 🔹 Categorical Analysis: `Embarked`, `Sex`, `Pclass`
```python
df['Embarked'].value_counts().plot(kind='bar')
sns.heatmap(pd.crosstab(df['Survived'], df['Pclass'], normalize='columns') * 100)
```

---

## ⚠️ Challenges & Solutions

| Challenge                             | Strategy |
|--------------------------------------|----------|
| 🧩 **Missing values** in `Age`, `Cabin` | Calculated % missing, filled `Cabin` with placeholder `M`, explored grouping titles for imputing Age |
| 📐 **Skewed Fare distribution**       | Applied log transformation and derived `individual_fare` |
| 💡 **Sparse `Ticket`/`Cabin` fields** | Extracted deck level from `Cabin`; ignored highly fragmented `Ticket` |
| 🧠 **Unstructured Name field**        | Extracted `title`, grouped rare titles under 'other' |

---

## 🔍 Key Insights

- **Sex**: Females had ~74% survival rate, males only ~19%
- **Pclass**: Higher class → higher chances of survival
- **Family**: Derived `family_size` and `family_type`; small families had best odds
- **Cabin Deck**: Deck B > Deck C > others in survival rates
- **Age**: Children and young adults had modest survival advantage

---

## 🏁 Conclusion

This EDA provides a holistic view of survival factors aboard the Titanic. Through detailed analysis and thoughtful feature engineering, this project lays a robust groundwork for predictive modeling and showcases a full-cycle data analyst approach—from raw data to actionable story.

---

Feel free to tweak the tone or structure depending on your repo theme. Want me to help you polish the notebook or suggest a commit message strategy?
