"""Database operations for the Agriculture RAG Chatbot."""
import re
import pymssql
import pandas as pd
from typing import Optional, List, Dict, Any
from config import DB_CONFIG, ALLOWED_TABLES


class Database:
    """Manages database connections and queries."""

    def __init__(self):
        self._connection = None
        self._allowed_tables = [t.lower() for t in ALLOWED_TABLES]

    @property
    def connection(self):
        """Lazy connection to database."""
        if self._connection is None:
            try:
                self._connection = pymssql.connect(
                    server=DB_CONFIG["host"],
                    port=DB_CONFIG["port"],
                    user=DB_CONFIG["user"],
                    password=DB_CONFIG["password"],
                    database=DB_CONFIG["database"],
                    as_dict=True
                )
                print(f"[DB] Connected to {DB_CONFIG['database']}")
            except Exception as e:
                print(f"[DB] Connection failed: {e}")
                raise
        return self._connection

    def _validate_query(self, query: str) -> None:
        """Validate query only accesses allowed tables."""
        query_lower = query.lower()
        tables = re.findall(r'from\s+\[?(\w+)\]?', query_lower)
        tables += re.findall(r'join\s+\[?(\w+)\]?', query_lower)

        for table in tables:
            if table not in self._allowed_tables:
                raise PermissionError(f"Access denied to table: {table}")

    def execute(self, query: str) -> pd.DataFrame:
        """Execute a SELECT query and return results as DataFrame."""
        self._validate_query(query)

        cursor = self.connection.cursor()
        cursor.execute(query)
        results = cursor.fetchall()
        return pd.DataFrame(results) if results else pd.DataFrame()

    def get_schema(self) -> str:
        """Get schema for all allowed tables."""
        schema_parts = []
        cursor = self.connection.cursor()

        for table in ALLOWED_TABLES:
            cursor.execute(f"""
                SELECT COLUMN_NAME, DATA_TYPE
                FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_NAME = '{table}'
            """)
            cols = cursor.fetchall()
            if cols:
                schema_parts.append(f"-- {table}")
                for col in cols:
                    schema_parts.append(f"   {col['COLUMN_NAME']} ({col['DATA_TYPE']})")

        return "\n".join(schema_parts)

    def get_all_species(self) -> List[Dict[str, Any]]:
        """Get all species with their details."""
        df = self.execute("""
            SELECT SpeciesID, Species, MaleTerm, FemaleTerm, BabyTerm,
                   SingularTerm, PluralTerm, GestationPeriod
            FROM Speciesavailable
            WHERE SpeciesAvailable = 1
        """)
        return df.to_dict('records') if not df.empty else []

    def get_breeds_for_species(self, species_id: int) -> List[Dict[str, Any]]:
        """Get all breeds for a specific species."""
        df = self.execute(f"""
            SELECT b.BreedLookupID, b.Breed, b.Breeddescription,
                   b.MeatBreed, b.MilkBreed, b.WoolBreed, b.EggBreed, b.Working,
                   s.Species
            FROM Speciesbreedlookuptable b
            JOIN Speciesavailable s ON b.SpeciesID = s.SpeciesID
            WHERE b.SpeciesID = {species_id} AND b.breedavailable = 1
        """)
        return df.to_dict('records') if not df.empty else []

    def get_all_breeds(self) -> List[Dict[str, Any]]:
        """Get all breeds with species info."""
        df = self.execute("""
            SELECT TOP 2000 b.BreedLookupID, b.Breed, b.Breeddescription,
                   b.MeatBreed, b.MilkBreed, b.WoolBreed, b.EggBreed, b.Working,
                   s.Species, s.SpeciesID
            FROM Speciesbreedlookuptable b
            JOIN Speciesavailable s ON b.SpeciesID = s.SpeciesID
            WHERE b.breedavailable = 1
        """)
        return df.to_dict('records') if not df.empty else []

    def get_colors_for_species(self, species_id: int) -> List[str]:
        """Get available colors for a species."""
        df = self.execute(f"""
            SELECT DISTINCT SpeciesColor
            FROM Speciescolorlookuptable
            WHERE SpeciesID = {species_id}
        """)
        return df['SpeciesColor'].tolist() if not df.empty else []

    def get_patterns_for_species(self, species_id: int) -> List[str]:
        """Get available patterns for a species."""
        df = self.execute(f"""
            SELECT DISTINCT SpeciesColor as Pattern
            FROM Speciespatternlookuptable
            WHERE SpeciesID = {species_id}
        """)
        return df['Pattern'].tolist() if not df.empty else []

    def get_categories_for_species(self, species_id: int) -> List[str]:
        """Get categories for a species."""
        df = self.execute(f"""
            SELECT SpeciesCategory
            FROM Speciescategory
            WHERE SpeciesID = {species_id}
            ORDER BY SpeciesCategoryOrder
        """)
        return df['SpeciesCategory'].tolist() if not df.empty else []

    def search_breeds(self, search_term: str) -> List[Dict[str, Any]]:
        """Search breeds by name."""
        df = self.execute(f"""
            SELECT TOP 20 b.Breed, b.Breeddescription, s.Species,
                   b.MeatBreed, b.MilkBreed, b.WoolBreed, b.EggBreed
            FROM Speciesbreedlookuptable b
            JOIN Speciesavailable s ON b.SpeciesID = s.SpeciesID
            WHERE b.Breed LIKE '%{search_term}%' AND b.breedavailable = 1
        """)
        return df.to_dict('records') if not df.empty else []

    def get_database_summary(self) -> Dict[str, Any]:
        """Get a summary of database contents."""
        summary = {}

        # Count species
        df = self.execute("SELECT COUNT(*) as cnt FROM Speciesavailable WHERE SpeciesAvailable = 1")
        summary['total_species'] = df['cnt'].iloc[0] if not df.empty else 0

        # Count breeds
        df = self.execute("SELECT COUNT(*) as cnt FROM Speciesbreedlookuptable WHERE breedavailable = 1")
        summary['total_breeds'] = df['cnt'].iloc[0] if not df.empty else 0

        # Count colors
        df = self.execute("SELECT COUNT(DISTINCT SpeciesColor) as cnt FROM Speciescolorlookuptable")
        summary['total_colors'] = df['cnt'].iloc[0] if not df.empty else 0

        # Count patterns
        df = self.execute("SELECT COUNT(DISTINCT SpeciesColor) as cnt FROM Speciespatternlookuptable")
        summary['total_patterns'] = df['cnt'].iloc[0] if not df.empty else 0

        return summary


# Singleton instance
db = Database()
