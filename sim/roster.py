"""Fictional squads (same fictional players as the original dataset) + formation slots."""
# (id, name, position, jersey)
KES = [(91001, "Anselm Rovik", "Goalkeeper", 1), (91002, "Kofi Danlade", "Right Back", 2), (91003, "Luka Brandvold", "Right Center Back", 4),
       (91004, "Emil Sarrazin", "Left Center Back", 5), (91005, "Joaquin Pellerano", "Left Back", 3),
       (91006, "Idris Tavernier", "Right Defensive Midfield", 6), (91007, "Niko Halvarsen", "Left Defensive Midfield", 8),
       (91008, "Samir Okonjo", "Right Wing", 7), (91009, "Teo Marchetti", "Center Attacking Midfield", 10),
       (91010, "Yuto Kasamatsu", "Left Wing", 11), (91011, "Mateo Valdrin", "Center Forward", 9)]
KES_SUBS = [(91012, "Dmitri Valcourt", "Right Wing", 14), (91013, "Bram Eskelinen", "Left Defensive Midfield", 16)]
IRO = [(92001, "Oskar Brennvall", "Goalkeeper", 1), (92002, "Fabio Santarelli", "Right Back", 2), (92003, "Gideon Achterberg", "Right Center Back", 4),
       (92004, "Marek Dostal", "Left Center Back", 5), (92005, "Ismail Bakary", "Left Back", 3), (92006, "Callum Strathern", "Right Midfield", 7),
       (92007, "Jonas Wirtanen", "Right Center Midfield", 8), (92008, "Leandro Ferrez", "Left Center Midfield", 6),
       (92009, "Tarik Mbemba", "Left Midfield", 11), (92010, "Dario Oquendo", "Right Center Forward", 9), (92011, "Stellan Kriek", "Left Center Forward", 10)]
IRO_SUBS = [(92012, "Wesley Nakamura", "Right Midfield", 14), (92013, "Noel Abernathy", "Left Center Midfield", 17)]

POS_ID = {"Goalkeeper": 1, "Right Back": 2, "Right Center Back": 3, "Left Center Back": 5, "Left Back": 6, "Right Defensive Midfield": 9,
          "Left Defensive Midfield": 11, "Right Midfield": 12, "Right Center Midfield": 13, "Left Center Midfield": 15, "Left Midfield": 16,
          "Right Wing": 17, "Left Wing": 21, "Center Attacking Midfield": 18, "Center Forward": 23, "Right Center Forward": 22, "Left Center Forward": 24}

# slot (attacking frame: x towards opponent goal, y 0..80, "Right" = high y as in the original data)
SLOT = {"Goalkeeper": (6, 40), "Right Back": (30, 68), "Right Center Back": (24, 50), "Left Center Back": (24, 30), "Left Back": (30, 12),
        "Right Defensive Midfield": (46, 48), "Left Defensive Midfield": (46, 32), "Right Midfield": (58, 68), "Right Center Midfield": (54, 48),
        "Left Center Midfield": (54, 32), "Left Midfield": (58, 12), "Right Wing": (80, 66), "Left Wing": (80, 14),
        "Center Attacking Midfield": (72, 40), "Center Forward": (92, 40), "Right Center Forward": (88, 48), "Left Center Forward": (88, 32)}


def role_of(pos: str) -> str:
    if pos == "Goalkeeper":
        return "GK"
    if "Back" in pos:
        return "DEF"
    if "Wing" in pos or "Attacking" in pos or "Forward" in pos:
        return "ATT"
    return "MID"
