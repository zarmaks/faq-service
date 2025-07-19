"""
Script to fix the database schema by adding the missing 'source' column.
"""

import sqlite3
import os

def fix_database():
    """Add the missing 'source' column to the questions table."""
    db_path = "faq.db"
    
    if not os.path.exists(db_path):
        print("❌ Database file not found")
        return False
    
    try:
        # Connect to the database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if the source column exists
        cursor.execute("PRAGMA table_info(questions)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if 'source' in columns:
            print("✅ 'source' column already exists")
            return True
        
        # Add the missing column
        print("🔧 Adding 'source' column to questions table...")
        cursor.execute("ALTER TABLE questions ADD COLUMN source VARCHAR(50) DEFAULT 'api'")
        
        # Commit the changes
        conn.commit()
        print("✅ Database schema fixed successfully!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error fixing database: {e}")
        return False
        
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    fix_database()
