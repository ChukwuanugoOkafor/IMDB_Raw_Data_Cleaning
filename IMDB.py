import pandas as pd  # Library for data manipulation and analysis
from sklearn.linear_model import LinearRegression  # Linear regression model from sklearn

# Configure pandas to display all columns, rows, and full content for better visibility
pd.set_option('display.max_columns', None)  # Show all columns in DataFrame
pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_colwidth', None)  # Show full content in each cell
pd.set_option('display.width', None)  # Allow wide DataFrame to be displayed fully

# Set the file path for loading the dataset
file_path = "C:\\Users\\ABC\\Desktop\\Real data\\Imdb_top_1000.csv"

# Load the CSV file into a pandas DataFrame
df_IMDB_raw_file = pd.read_csv(file_path)
file_Name = "Imdb_top_1000.csv"

# Print first 5 rows to verify data has loaded correctly
print(f"\n✅ Displaying the first five rows of {file_Name} to verify that the dataset is accessible and to have a clue of the contents")
print(df_IMDB_raw_file.head())

# Convert 'Released_Year' column to numeric, coercing errors to NaN, then cast to Int64 (nullable int)
df_IMDB_raw_file["Released_Year"] = pd.to_numeric(df_IMDB_raw_file["Released_Year"], errors='coerce').astype("Int64")
# Drop rows with missing 'Released_Year'
df_IMDB_raw_file = df_IMDB_raw_file.dropna(subset=["Released_Year"])

# Clean 'Runtime' column by removing ' min' and converting to integer
df_IMDB_raw_file['Runtime'] = df_IMDB_raw_file['Runtime'].str.replace(' min', '').astype(int)
# Rename the cleaned runtime column
df_IMDB_raw_file.rename(columns={'Runtime': 'Runtime_in_minutes'}, inplace=True)

# --- Normalize the Genre column using one-hot encoding ---
genre_index = df_IMDB_raw_file.columns.get_loc('Genre')  # Get index of Genre column
# Convert Genre string into dummy/one-hot encoded DataFrame
genre_dummies = df_IMDB_raw_file['Genre'].str.get_dummies(sep=', ')
# Rename dummy columns for clarity
genre_dummies.columns = ['Genre_' + col for col in genre_dummies.columns]
# Insert genre dummy columns into the main DataFrame at the correct position
for i, col in enumerate(genre_dummies.columns):
    df_IMDB_raw_file.insert(genre_index + i, col, genre_dummies[col])

# --- Predict missing Meta_score values using linear regression ---
# Split the data into known and missing Meta_score subsets
df_known = df_IMDB_raw_file[df_IMDB_raw_file['Meta_score'].notnull()]
df_missing = df_IMDB_raw_file[df_IMDB_raw_file['Meta_score'].isnull()]

# Define features for predicting Meta_score
features = ['IMDB_Rating', 'Runtime_in_minutes'] + genre_dummies.columns.tolist()
X_train = df_known[features]  # Features for training
y_train = df_known['Meta_score']  # Target variable
model = LinearRegression()  # Initialize linear regression model
model.fit(X_train, y_train)  # Train model on known data

X_missing = df_missing[features]  # Features for prediction
predicted_scores = model.predict(X_missing)  # Predict missing Meta_scores
# Replace missing Meta_score values with predicted scores
df_IMDB_raw_file.loc[df_IMDB_raw_file['Meta_score'].isnull(), 'Meta_score'] = predicted_scores

# Clean 'Gross' column by removing commas and converting to float
df_IMDB_raw_file['Gross'] = df_IMDB_raw_file['Gross'].str.replace(',', '', regex=False).astype(float)
pd.options.display.float_format = '{:,.2f}'.format  # Format float display

# --- Predict missing Gross values using linear regression ---
# Define features for predicting Gross
Pfeatures = ['IMDB_Rating', 'Meta_score', 'Runtime_in_minutes', 'Released_Year', 'No_of_Votes'] + genre_dummies.columns.tolist()
# Prepare training data where Gross is known
train_data = df_IMDB_raw_file[df_IMDB_raw_file['Gross'].notnull()].dropna(subset=Pfeatures)
X_train = train_data[Pfeatures]
y_train = train_data['Gross']

# Prepare data where Gross is missing
predict_data = df_IMDB_raw_file[df_IMDB_raw_file['Gross'].isnull()].dropna(subset=Pfeatures)
X_predict = predict_data[Pfeatures]

# Train model and predict missing Gross values
model = LinearRegression()
model.fit(X_train, y_train)
predicted_values = model.predict(X_predict)

# Fill predicted Gross values into original DataFrame
df_IMDB_raw_file.loc[df_IMDB_raw_file['Gross'].isnull(), 'Gross'] = predicted_values

# Print first 5 rows to confirm formatting and imputations
print(f"\n✅ Displaying the first five rows of {file_Name} to verify the formating implementation")
print(df_IMDB_raw_file.head())

# Display structure and data types of the DataFrame
print(f"\n✅ Displaying basic information about the dataset: {file_Name}")
df_IMDB_raw_file.info()

# Fill missing certificates with "Unknown"
print(f"\n✅ Handling missing values in the dataset: {file_Name}")
df_IMDB_raw_file['Certificate'] = df_IMDB_raw_file['Certificate'].fillna("Unknown")

# Display again after filling missing values
print(f"\n✅ Displaying the first five rows of {file_Name} to verify the formating implementation")
print(df_IMDB_raw_file.head())
print(f"\n✅ Displaying basic information about the dataset: {file_Name} after handling missing values")
df_IMDB_raw_file.info()

# Show descriptive statistics of the dataset
print(f"\n✅ Displaying basic statistics about the dataset: {file_Name}\n", df_IMDB_raw_file.describe())

# Check for and remove duplicate rows
print(f"\n✅ Duplicate Rows in the Dataset: {file_Name}")
print(f"The number of duplicate rows in the {file_Name} is:", df_IMDB_raw_file.duplicated().sum())
if df_IMDB_raw_file.duplicated().sum() > 0:
    duplicate_rows = df_IMDB_raw_file[df_IMDB_raw_file.duplicated()]
    print(f"Duplicate Rows:\n{duplicate_rows}")
    df_IMDB_raw_file.drop_duplicates(inplace=True)
    print(f"✅ Duplicated rows in {file_Name} has been dropped")
else:
    print("✅ No duplicate rows found.")

# Check for rows with NaN or empty string
print(f"\nRows with NaN or Empty string in the Dataset: {file_Name}")
nan_or_empty_count = (df_IMDB_raw_file.isnull().any(axis=1) | (df_IMDB_raw_file == '').any(axis=1)).sum()
print(nan_or_empty_count)

if nan_or_empty_count > 0:
    rows_with_nan_or_empty = df_IMDB_raw_file[df_IMDB_raw_file.isnull().any(axis=1) | (df_IMDB_raw_file == '').any(axis=1)]
    print(f"\nRows with NaN values or empty cells in {file_Name}:\n", rows_with_nan_or_empty)

# Create movies table with selected fields
movies_df = df_IMDB_raw_file[['Series_Title', 'Released_Year', 'Runtime_in_minutes', 'IMDB_Rating', 'Meta_score', 'Gross', 'Certificate', 'Poster_Link']].copy()
movies_df['MovieID'] = movies_df.reset_index().index + 1  # Add unique MovieID

# Create genre table
df_genre = df_IMDB_raw_file[['Series_Title', 'Genre']].copy()
df_genre['Genre'] = df_genre['Genre'].astype(str)
df_genre = df_genre.assign(Genre=df_genre['Genre'].str.split(', ')).explode('Genre')  # Explode genre list

# Create unique genre list with IDs
unique_genres = pd.DataFrame(df_genre['Genre'].unique(), columns=['Genre'])
unique_genres['GenreID'] = unique_genres.index + 1

# Map genres to movies
df_genre = df_genre.merge(unique_genres, on='Genre')
df_genre = df_genre.merge(movies_df[['Series_Title', 'MovieID']], on='Series_Title')
movie_genres = df_genre[['MovieID', 'GenreID']]

# Create director table
df_director = df_IMDB_raw_file[['Series_Title', 'Director']].copy()
unique_directors = pd.DataFrame(df_director['Director'].unique(), columns=['Director'])
unique_directors['DirectorID'] = unique_directors.index + 1

# Map directors to movies
df_director = df_director.merge(unique_directors, on='Director')
df_director = df_director.merge(movies_df[['Series_Title', 'MovieID']], on='Series_Title')
movie_directors = df_director[['MovieID', 'DirectorID']]

# Normalise stars
stars_df = df_IMDB_raw_file[['Series_Title', 'Star1', 'Star2', 'Star3', 'Star4']].copy()
stars_long = stars_df.melt(id_vars='Series_Title', value_name='Star').drop('variable', axis=1).dropna()

# Create unique stars table
unique_stars = pd.DataFrame(stars_long['Star'].unique(), columns=['Star'])
unique_stars['StarID'] = unique_stars.index + 1

# Map stars to movies
stars_long = stars_long.merge(unique_stars, on='Star')
stars_long = stars_long.merge(movies_df[['Series_Title', 'MovieID']], on='Series_Title')
movie_stars = stars_long[['MovieID', 'StarID']]

# Drop original 'Genre' column since it's now normalized
df_IMDB_raw_file.drop('Genre', axis=1, inplace=True)

# Export all cleaned and normalised data to an Excel file
with pd.ExcelWriter("C:\\Users\\ABC\\Desktop\\Real data\\IMDB_Normalised_Final.xlsx") as writer:
    movies_df.to_excel(writer, sheet_name='Movies', index=False)
    unique_genres.to_excel(writer, sheet_name='Genres', index=False)
    movie_genres.to_excel(writer, sheet_name='Movie_Genres', index=False)
    unique_directors.to_excel(writer, sheet_name='Directors', index=False)
    movie_directors.to_excel(writer, sheet_name='Movie_Directors', index=False)
    unique_stars.to_excel(writer, sheet_name='Stars', index=False)
    movie_stars.to_excel(writer, sheet_name='Movie_Stars', index=False)
