class DatabaseClient:
    def __init__(self, connection_string):
        self.conn = connection_string
        print(f"Connected to DB at {connection_string}")

    def fetch_user_profile(self, user_id):
        """Retrieves user profile data from DB."""
        return {"user_id": user_id, "name": "Alice", "role": "Developer"}