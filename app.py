import streamlit as st
import my_module

def main():
    st.title("Simple Streamlit App")

    # Using a function from another Python script
    result = my_module.add_numbers(5, 7)
    st.write("Result of adding 5 and 7:", result)

if __name__ == "__main__":
    main()