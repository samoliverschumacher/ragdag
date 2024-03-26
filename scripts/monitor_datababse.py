
# Create an in-memory SQLite database
import sqlite3

from app import ROOT_DIR

path = str(ROOT_DIR.parent / 'sample.db')

print(path)
conn = sqlite3.connect(path)
cursor = conn.cursor()

# Create the 'eval' table
cursor.execute("""
    CREATE TABLE eval (
        id INTEGER PRIMARY KEY,
        event_id VARCHAR,
        log JSON
    )
""")

conn.commit()

conn.close()
