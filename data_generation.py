import numpy as np
import pandas as pd

# Zufallsgenerator für Reproduzierbarkeit
np.random.seed(42)

# Anzahl der Datensätze
n_samples = 5000

# Alter der Versicherten (18 bis 80 Jahre)
age = np.random.randint(18, 81, size=n_samples)

# Geschlecht (0 = weiblich, 1 = männlich)
gender = np.random.choice([0, 1], size=n_samples)

# Fahrpraxis in Jahren (0 bis Alter - 18)
driving_experience = age - np.random.randint(16, 21, size=n_samples)
driving_experience = np.clip(driving_experience, 0, None)

# Fahrzeugtyp (0 = Kleinwagen, 1 = Limousine, 2 = SUV, 3 = Sportwagen)
vehicle_type = np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.4, 0.3, 0.2, 0.1])

# Vorherige Unfälle (Poisson-Verteilung)
previous_accidents = np.random.poisson(0.1, size=n_samples)

# Wohnort (0 = ländlich, 1 = städtisch)
region = np.random.choice([0, 1], size=n_samples)

# Jahreskilometerleistung (5.000 bis 30.000 km)
annual_mileage = np.random.normal(15000, 5000, size=n_samples)
annual_mileage = np.clip(annual_mileage, 5000, 30000)

# Versicherungsprämie (200€ bis 1000€)
premium = np.random.normal(500, 100, size=n_samples)
premium = np.clip(premium, 200, 1000)

# Risikoberechnung für Schadensfall
risk_score = (
    (80 - age) * 0.02 +                # Jüngere Fahrer höheres Risiko
    (vehicle_type == 3) * 0.05 +       # Sportwagen höheres Risiko
    (region == 1) * 0.03 +             # Städtische Gebiete höheres Risiko
    previous_accidents * 0.1 +         # Vorherige Unfälle erhöhen Risiko
    (annual_mileage / 10000) * 0.02    # Mehr Kilometer erhöhen Risiko
)

# Wahrscheinlichkeit für Schadensfall (Sigmoid-Funktion)
prob_accident = 1 / (1 + np.exp(-risk_score))

# Schadensfall (0 = Nein, 1 = Ja)
accident = np.random.binomial(1, prob_accident)

# DataFrame erstellen
data = pd.DataFrame({
    'Age': age,
    'Gender': gender,
    'Driving_Experience': driving_experience,
    'Vehicle_Type': vehicle_type,
    'Previous_Accidents': previous_accidents,
    'Region': region,
    'Annual_Mileage': annual_mileage,
    'Premium': premium,
    'Accident': accident
})

# Daten in CSV-Datei speichern
data.to_csv('insurance_data.csv', index=False)

print("Datensatz 'insurance_data.csv' wurde erfolgreich erstellt.")
