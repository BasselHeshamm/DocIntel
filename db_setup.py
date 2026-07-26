import sqlite3

# connect() opens (or creates, if it doesn't exist yet) a database file on disk.
# Unlike PostgreSQL, there's no server to start -- this file IS the whole database.
conn = sqlite3.connect("docintel.db")

# A cursor is the object you use to actually execute SQL commands through the connection.
cursor = conn.cursor()

# CREATE TABLE defines the shape of our data, like the InvoiceFields pydantic
# class did for the LLM -- except this shape is now permanent, on disk.
cursor.execute("""
CREATE TABLE IF NOT EXISTS extractions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    raw_text TEXT,
    vendor_name TEXT,
    vendor_source TEXT,
    total_amount TEXT,
    total_amount_source TEXT,
    invoice_date TEXT,
    invoice_date_source TEXT,
    invoice_number TEXT,
    invoice_number_source TEXT
)
""")

# Changes to a database aren't saved until you explicitly "commit" them --
# this is a safety feature, so partial/broken writes don't corrupt your data.
conn.commit()
conn.close()

print("Database and table created: docintel.db")