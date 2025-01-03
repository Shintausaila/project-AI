import streamlit as st
import pandas as pd
import streamlit as st
import pandas as pd

def load_data():
    data = pd.read_csv('hotel_reviews.csv')
    return data

data = load_data()

def main():
    st.title('Hotel Reviews Viewer')
    
    st.write(data)

    # Visualisasi data (opsional)
    if st.checkbox('Show Histogram of Ratings'):
        st.subheader('Histogram of Ratings')
        st.bar_chart(data['Rating'].value_counts())

if __name__ == '__main__':
    main()
