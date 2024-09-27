# database.py
import sqlite3
import numpy as np
# Connect to SQLite database
conn = sqlite3.connect('criminal_db.db')
c = conn.cursor()

# Create the table for criminals if it doesn't exist
def initialize_database():
    c.execute('''CREATE TABLE IF NOT EXISTS criminals
                 (name TEXT, encoding BLOB)''')
    conn.commit()

# Store name and face encoding into the database
def store_in_database(name, face_encoding):
    encoding_blob = face_encoding.tobytes()
    c.execute("INSERT INTO criminals (name, encoding) VALUES (?, ?)", (name, encoding_blob))
    conn.commit()
    print("added successfully")

# Retrieve all criminal records (name and encoding) from the database
def get_all_criminals():
    c.execute("SELECT * FROM criminals")
    return c.fetchall()
