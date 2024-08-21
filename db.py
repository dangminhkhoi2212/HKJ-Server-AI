from supabase import create_client, Client

# Supabase setup
url = "https://ckduvpcvhlbgrujnrftb.supabase.co"
key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImNrZHV2cGN2aGxiZ3J1am5yZnRiIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTcyMzQ2MDI2MywiZXhwIjoyMDM5MDM2MjYzfQ.Kwa1bfhFeGsqpDzHGzYLwYdYPRebAV6R5JwOXTkPWUs"
supabase: Client = create_client(url, key)
