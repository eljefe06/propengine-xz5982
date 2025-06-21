import os
import argparse
from datetime import datetime, timedelta
from typing import List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from rapidfuzz import process, fuzz
from scipy.stats import poisson, nbinom
from pybaseball import pitching_stats_bref

# Default Odds API key if not provided via environment
os.environ.setdefault("ODDS_API_KEY", "17c146974ce65af7fd9300139a2ceb4e")

# Factores de parque simples (1.0 = neutral)
PARK_FACTORS = {
    "Arizona Diamondbacks": 1.04,
    "Atlanta Braves": 0.98,
    "Baltimore Orioles": 0.96,
    "Boston Red Sox": 1.04,
    "Chicago Cubs": 0.99,
    "Chicago White Sox": 1.02,
    "Cincinnati Reds": 1.03,
    "Cleveland Guardians": 0.96,
    "Colorado Rockies": 1.15,
    "Detroit Tigers": 0.97,
    "Houston Astros": 0.97,
    "Kansas City Royals": 1.02,
    "Los Angeles Angels": 1.01,
    "Los Angeles Dodgers": 0.98,
    "Miami Marlins": 0.99,
    "Milwaukee Brewers": 1.00,
    "Minnesota Twins": 1.04,
    "New York Mets": 0.99,
    "New York Yankees": 0.98,
    "Oakland Athletics": 0.96,
    "Philadelphia Phillies": 1.01,
    "Pittsburgh Pirates": 0.98,
    "San Diego Padres": 0.97,
    "San Francisco Giants": 0.95,
    "Seattle Mariners": 0.98,
    "St. Louis Cardinals": 1.02,
    "Tampa Bay Rays": 0.98,
    "Texas Rangers": 1.03,
    "Toronto Blue Jays": 0.99,
    "Washington Nationals": 1.01,
}

# Factores de umpire en tasa de ponches (1.0 = neutral)
UMP_K_FACTORS = {
    "Pat Hoberg": 1.08,
    "Chris Guccione": 1.05,
    "Laz Diaz": 0.92,
    "Angel Hernandez": 0.95,
}


def mlb_next_games_date_local(cutoff_hour: int = 22) -> str:
    """Return today's date if games remain, otherwise the next date with MLB games."""
    tz = datetime.now().astimezone().tzinfo
    now = datetime.now(tz)
    today_str = now.strftime("%Y-%m-%d")
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={today_str}"
    try:
        res = requests.get(url, timeout=10)
        data = res.json()
    except Exception:
        return today_str
    fechas = data.get("dates", [])
    if fechas and fechas[0].get("games"):
        for game in fechas[0]["games"]:
            game_time_utc = game["gameDate"]
            game_time = datetime.fromisoformat(game_time_utc.replace("Z", "+00:00")).astimezone(tz)
            if game_time > now:
                return today_str
    for i in range(1, 7):
        next_date = (now.date() + timedelta(days=i)).strftime("%Y-%m-%d")
        url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={next_date}"
        try:
            res = requests.get(url, timeout=10)
            data = res.json()
        except Exception:
            continue
        fechas = data.get("dates", [])
        if fechas and fechas[0].get("games"):
            return next_date
    raise RuntimeError("No hay juegos MLB en los próximos días.")


# 1. Obtener abridores probables desde MLB.com para una fecha dada

def get_probable_pitchers(date: str) -> pd.DataFrame:
    """Devuelve abridores probables junto con IDs y árbitro principal."""
    url = (
        "https://statsapi.mlb.com/api/v1/schedule"
        f"?sportId=1&date={date}&hydrate=probablePitcher,officials"
    )
    try:
        res = requests.get(url, timeout=10)
    except Exception:
        return pd.DataFrame()
    if res.status_code != 200:
        raise RuntimeError(f"No se pudo obtener datos de MLB.com. Codigo {res.status_code}")
    data = res.json()
    fechas = data.get("dates", [])
    if not fechas:
        return pd.DataFrame()

    probables = []
    for juego in fechas[0].get("games", []):
        equipos = juego["teams"]
        officials = {
            o["officialType"]: o["official"]["fullName"]
            for o in juego.get("officials", [])
        }
        hp_ump = officials.get("Home Plate")
        for lado, contrario in [("home", "away"), ("away", "home")]:
            pitcher = equipos[lado].get("probablePitcher")
            if pitcher and pitcher.get("fullName"):
                probables.append(
                    {
                        "team": equipos[lado]["team"]["name"],
                        "team_id": equipos[lado]["team"]["id"],
                        "pitcher": pitcher.get("fullName"),
                        "pitcher_id": pitcher.get("id"),
                        "opponent": equipos[contrario]["team"]["name"],
                        "opponent_id": equipos[contrario]["team"]["id"],
                        "game_pk": juego.get("gamePk"),
                        "umpire": hp_ump,
                    }
                )
    return pd.DataFrame(probables)


# 2. Estadísticas anuales desde Baseball Reference con respaldo a MLB Stats

def fetch_stats_mlb_api(year: int) -> pd.DataFrame:
    """Fallback using MLB Stats API if Baseball Reference fails."""
    url = "https://statsapi.mlb.com/api/v1/stats"
    params = {
        "stats": "season",
        "group": "pitching",
        "season": year,
        "playerPool": "all",
    }
    try:
        res = requests.get(url, params=params, timeout=10)
    except Exception:
        return pd.DataFrame()
    if res.status_code != 200:
        print("⚠️ Error MLB Stats API", res.status_code)
        return pd.DataFrame()
    rows = []
    for s in res.json().get("stats", [{}])[0].get("splits", []):
        stat = s.get("stat", {})
        player = s.get("player", {})
        def to_float(val):
            try:
                return float(val)
            except (TypeError, ValueError):
                return None
        nombre = player.get("fullName")
        if not nombre:
            continue
        rows.append({
            "FullName": nombre,
            "mlbID": player.get("id"),
            "W": stat.get("wins"),
            "L": stat.get("losses"),
            "ERA": to_float(stat.get("era")),
            "GS": stat.get("gamesStarted"),
            "IP": to_float(stat.get("inningsPitched")),
            "SO": stat.get("strikeOuts"),
            "BB": stat.get("baseOnBalls"),
            "SO/W": to_float(stat.get("strikeoutWalkRatio")),
            "WHIP": to_float(stat.get("whip")),
            "SO9": to_float(stat.get("strikeoutsPer9Inn")),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df[df["FullName"].notnull()]
    return df


def get_pitching_stats(year: int) -> pd.DataFrame:
    """Retrieve season pitching stats, trying BRef then MLB Stats."""
    try:
        stats = pitching_stats_bref(year)
    except Exception as e:
        print(f"⚠️ Error obteniendo datos de Baseball Reference para {year}: {e}")
        stats = pd.DataFrame()
    if stats.empty and year > 1900:
        try:
            stats = pitching_stats_bref(year - 1)
        except Exception as e:
            print(f"⚠️ Error obteniendo datos del año previo {year-1}: {e}")
    if stats.empty:
        print("⚠️ Datos de Baseball Reference no disponibles, usando MLB Stats API")
        stats = fetch_stats_mlb_api(year)
    if "Name" in stats.columns:
        stats = stats.rename(columns={"Name": "FullName"})
    if not stats.empty:
        stats = stats[stats["FullName"].notnull()]
    return stats


# 3. Filtrar datos de estadisticas para solo los pitchers listados

import unicodedata

def _normalize_name(name: str) -> str:
    """Return a simplified string for fuzzy comparisons."""
    if not isinstance(name, str):
        return ""
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if not unicodedata.combining(c))
    name = (
        name.lower()
        .replace("jr.", "")
        .replace("jr", "")
        .replace("sr.", "")
        .replace("sr", "")
        .replace(".", "")
        .replace("-", " ")
        .replace("'", "")
    )
    return " ".join(name.split())


def filtrar_stats(df_stats: pd.DataFrame, prob_df: pd.DataFrame) -> pd.DataFrame:
    """Merge probable pitchers with season stats using IDs and fuzzy names."""
    if prob_df.empty:
        return pd.DataFrame()

    stats_by_id = {}
    if not df_stats.empty:
        for _, row in df_stats.iterrows():
            stats_by_id[row.get("mlbID")] = row

    def _normalize(name: str) -> str:
        return _normalize_name(name)

    rows = []
    for _, p_row in prob_df.iterrows():
        stat_row = None

        pid = p_row.get("pitcher_id")
        if pid in stats_by_id:
            stat_row = stats_by_id[pid]
        elif not df_stats.empty:
            norm_pit = _normalize(p_row["pitcher"])
            df_stats["norm"] = df_stats["FullName"].apply(_normalize)
            match = df_stats["norm"].eq(norm_pit)
            if match.any():
                stat_row = df_stats[match].iloc[0]
            else:
                result = process.extractOne(norm_pit, df_stats["norm"], scorer=fuzz.token_sort_ratio)
                if result and result[1] >= 80:
                    stat_row = df_stats.iloc[result[2]]

        if stat_row is not None:
            fila = {
                "FullName": p_row["pitcher"],
                "team": p_row["team"],
                "opponent": p_row["opponent"],
                "W": stat_row.get("W"),
                "L": stat_row.get("L"),
                "ERA": stat_row.get("ERA"),
                "GS": stat_row.get("GS"),
                "IP": stat_row.get("IP"),
                "SO9": stat_row.get("SO9"),
                "IP_por_GS": (stat_row.get("IP") / stat_row.get("GS")) if stat_row.get("GS") else None,
                "pitcher_id": p_row["pitcher_id"],
                "team_id": p_row["team_id"],
                "opponent_id": p_row["opponent_id"],
                "umpire": p_row.get("umpire"),
            }
        else:
            fila = {
                "FullName": p_row["pitcher"],
                "team": p_row["team"],
                "opponent": p_row["opponent"],
                "W": "Sin stats",
                "L": "Sin stats",
                "ERA": "Sin stats",
                "GS": "Sin stats",
                "IP": "Sin stats",
                "SO9": "Sin stats",
                "IP_por_GS": "Sin stats",
                "pitcher_id": p_row["pitcher_id"],
                "team_id": p_row["team_id"],
                "opponent_id": p_row["opponent_id"],
                "umpire": p_row.get("umpire"),
                "Sugerencia": "Sin datos suficientes para proyección",
            }
        rows.append(fila)

    result = pd.DataFrame(rows)
    result = result.drop_duplicates(subset=["FullName", "team", "opponent"])
    return result


# Estadísticas recientes del lanzador
def get_pitcher_logs(player_id: int, n: int = 5) -> pd.DataFrame:
    """Ultimas salidas del lanzador incluyendo hits y walks."""
    season = datetime.now().year
    params = {
        "stats": "gameLog",
        "group": "pitching",
        "season": season,
    }
    url = f"https://statsapi.mlb.com/api/v1/people/{player_id}/stats"
    try:
        res = requests.get(url, params=params, timeout=10)
    except Exception:
        return pd.DataFrame()
    if res.status_code != 200:
        return pd.DataFrame()
    stats_list = res.json().get("stats", [])
    if not stats_list:
        return pd.DataFrame()
    splits = stats_list[0].get("splits", [])[:n]
    registros = []
    for s in splits:
        stat = s.get("stat", {})
        ip = float(stat.get("inningsPitched", 0))
        so = stat.get("strikeOuts", 0)
        hits = stat.get("hits", 0)
        walks = stat.get("baseOnBalls", 0)
        fecha = s.get("date") or s.get("gameDate", "").split("T")[0]
        registros.append({"IP": ip, "SO": so, "H": hits, "BB": walks, "date": fecha})
    return pd.DataFrame(registros)


def streak_SO_over(logs: pd.DataFrame, line: float) -> Optional[str]:
    """Return streak of over/under strikeouts for recent logs."""
    if logs.empty or line is None:
        return None
    over = (logs["SO"] > line).sum()
    under = (logs["SO"] < line).sum()
    return f"{over} over, {under} under"


# 4. Obtener ponches por juego de un equipo en un rango de fechas

def team_k_per_game(team_id, start_date, end_date):
    url = (
        f"https://statsapi.mlb.com/api/v1/teams/{team_id}/stats?"
        f"stats=byDateRange&group=hitting&startDate={start_date}&endDate={end_date}"
    )
    try:
        res = requests.get(url, timeout=10)
    except Exception:
        return None
    if res.status_code != 200:
        return None
    data = res.json()
    splits = data.get("stats", [{}])[0].get("splits", [])
    if not splits:
        return None
    stat = splits[0]["stat"]
    so = stat.get("strikeOuts")
    games = stat.get("gamesPlayed")
    return so / games if games else None


# 5. Promedio de ponches por juego de toda la liga en un rango de fechas

def league_k_per_game(start_date, end_date):
    try:
        teams = requests.get("https://statsapi.mlb.com/api/v1/teams?sportId=1", timeout=10).json().get("teams", [])
    except Exception:
        return None
    total_so = 0
    total_games = 0
    for t in teams:
        kpg = team_k_per_game(t["id"], start_date, end_date)
        if kpg is None:
            continue
        # recuperamos juegos y ponches otra vez
        url = (
            f"https://statsapi.mlb.com/api/v1/teams/{t['id']}/stats?"
            f"stats=byDateRange&group=hitting&startDate={start_date}&endDate={end_date}"
        )
        try:
            res = requests.get(url, timeout=10)
        except Exception:
            continue
        data = res.json()
        splits = data.get("stats", [{}])[0].get("splits", [])
        if not splits:
            continue
        stat = splits[0]["stat"]
        total_so += stat.get("strikeOuts", 0)
        total_games += stat.get("gamesPlayed", 0)
    return total_so / total_games if total_games else None


# Calcular factor de uso del bullpen en un rango de fechas
def bullpen_usage_factor(team_id: int, start_date: str, end_date: str) -> float:
    """Return a multiplier (>1 means bullpen overused, so starter may go deeper)."""
    url = (
        f"https://statsapi.mlb.com/api/v1/teams/{team_id}/stats?stats=byDateRange"
        f"&group=pitching&startDate={start_date}&endDate={end_date}"
    )
    try:
        res = requests.get(url, timeout=10)
    except Exception:
        return 1.0
    if res.status_code != 200:
        return 1.0
    splits = res.json().get("stats", [{}])[0].get("splits", [])
    if not splits:
        return 1.0
    stat = splits[0].get("stat", {})
    games = stat.get("gamesPlayed", 0)
    ip = float(stat.get("inningsPitched", 0))
    if not games:
        return 1.0
    extra = max(0.0, ip - 9 * games)
    extra_pg = extra / games
    return 1.0 + min(extra_pg / 3.0, 0.15)


# ------------------------
# Datos de alineaciones y bateadores

def get_starting_lineup(team_id: int, date: str) -> List[dict]:
    """Devuelve lista de jugadores de la alineacion titular si esta disponible."""
    url = (
        f"https://statsapi.mlb.com/api/v1/teams/{team_id}/roster"
        f"?rosterType=startingLineups&date={date}"
    )
    try:
        res = requests.get(url, timeout=10)
    except Exception:
        return []
    if res.status_code != 200:
        return []
    return res.json().get("roster", [])


def get_batter_k_rate(player_id: int) -> Optional[float]:
    url = f"https://statsapi.mlb.com/api/v1/people/{player_id}/stats"
    params = {"stats": "season", "group": "hitting"}
    try:
        res = requests.get(url, params=params, timeout=10)
    except Exception:
        return None
    if res.status_code != 200:
        return None
    data = res.json()
    stats_list = data.get("stats", [])
    if not stats_list:
        return None
    splits = stats_list[0].get("splits", [])
    if not splits:
        return None
    stat = splits[0].get("stat", {})
    pa = stat.get("plateAppearances") or 0
    so = stat.get("strikeOuts") or 0
    return so / pa if pa else None


def lineup_k_rate(team_id: int, date: str) -> Optional[float]:
    players = get_starting_lineup(team_id, date)
    if not players:
        return None
    rates = []
    with ThreadPoolExecutor(max_workers=5) as ex:
        futures = {ex.submit(get_batter_k_rate, p["person"]["id"]): p for p in players}
        for fu in as_completed(futures):
            r = fu.result()
            if r is not None:
                rates.append(r)
    return sum(rates) / len(rates) if rates else None


# Obtener lineas de apuestas de ponches desde un Google Sheet compartido
def get_k_props(book: str = "FanDuel") -> pd.DataFrame:
    """Return strikeout lines from a shared odds sheet."""
    sheet_id = "1KwksrVfIhzT95I61s4tWgdfNfgMEsf1C_sISmFAhNow"
    tab = "Hoja 1"
    tab_url = tab.replace(" ", "%20")
    url = (
        f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={tab_url}"
    )
    try:
        odds_df = pd.read_csv(url)
    except Exception as e:
        print(f"⚠️ Error leyendo Google Sheet de odds: {e}")
        return pd.DataFrame()

    odds_df = odds_df[odds_df["market"] == "pitcher_strikeouts"]
    odds_overs = odds_df[
        (odds_df["label"].str.lower() == "over")
        & (odds_df["bookmaker"].str.lower() == book.lower())
    ].copy()
    if odds_overs.empty:
        return pd.DataFrame()

    odds_overs["price_num"] = (
        odds_overs["price"].astype(str).str.replace(",", ".").astype(float)
    )
    odds_overs["point_num"] = (
        odds_overs["point"].astype(str).str.replace(",", ".").astype(float)
    )

    best_odds = odds_overs.sort_values("price_num", ascending=False).drop_duplicates("description")
    best_odds = best_odds.rename(
        columns={
            "description": "pitcher",
            "point_num": "line",
            "price_num": "odds",
            "bookmaker": "book",
        }
    )
    return best_odds[["pitcher", "line", "odds", "book"]].reset_index(drop=True)


def match_prop(name: str, lines: pd.DataFrame) -> Optional[Tuple[pd.Series, int]]:
    if lines.empty:
        return None
    match, score, idx = process.extractOne(
        name, lines["pitcher"], scorer=fuzz.token_sort_ratio
    )
    if score < 70:
        return None
    serie = lines.iloc[idx]
    serie = serie.copy()
    serie["_match_score"] = score
    return serie, score


# 6. Calcular proyecciones ajustadas con rachas de rivales

def agregar_proyecciones(df, prob_df, lines_df, start_date, end_date):
    """Combina estadísticas y genera proyecciones y EV."""
    if df.empty:
        return df

    liga_kpg = league_k_per_game(start_date, end_date) or 8.5
    liga_k_rate = liga_kpg / 38  # aproximado de PAs por juego

    # Precalcular factores de bullpen por equipo
    bullpen_factors = {}
    for tid in df['team_id'].unique():
        bullpen_factors[tid] = bullpen_usage_factor(tid, start_date, end_date)
    df = df.merge(prob_df, left_on="FullName", right_on="pitcher", suffixes=("", "_prob"))
    if "team_prob" in df.columns:
        df["team"] = df["team_prob"]
        df.drop(columns=["team_prob"], inplace=True)
    if "opponent_prob" in df.columns:
        df["opponent"] = df["opponent_prob"]
        df.drop(columns=["opponent_prob"], inplace=True)

    for col in ["IP", "GS", "SO9"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    proys = []
    probs = []
    evs = []
    books = []
    odds_list = []
    lines = []
    suggestions = []
    confidences = []
    ev_validado = []
    streaks = []

    # Obtener logs en paralelo
    logs_map = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(get_pitcher_logs, pid): pid for pid in df["pitcher_id"]}
        for fu in as_completed(futs):
            logs_map[futs[fu]] = fu.result()

    for _, row in df.iterrows():
        lineup_rate = lineup_k_rate(row["opponent_id"], end_date)
        if lineup_rate is None:
            opp_kpg = team_k_per_game(row["opponent_id"], start_date, end_date) or liga_kpg
            factor = opp_kpg / liga_kpg
        else:
            factor = lineup_rate / liga_k_rate

        ip_pg = row["IP"] / row["GS"] if row.get("GS") else 0
        # Ajustar IP esperados segun uso reciente del bullpen
        bull_factor = bullpen_factors.get(row["team_id"], 1.0)
        ip_pg *= bull_factor
        base_proj = (row["SO9"] / 9) * ip_pg

        logs = logs_map.get(row.get("pitcher_id"), pd.DataFrame())
        n_logs = len(logs) if not logs.empty else 0
        weight_recent = 1.0 if n_logs >= 3 else 0.4 if n_logs == 2 else 0.15
        if not logs.empty and row["SO9"]:
            recent_k9 = (logs["SO"].sum() / logs["IP"].sum()) * 9 if logs["IP"].sum() else row["SO9"]
            base_proj *= (recent_k9 / row["SO9"]) * weight_recent + (1 - weight_recent)

        park_factor = PARK_FACTORS.get(row["team"], 1.0)
        ump_factor = UMP_K_FACTORS.get(row.get("umpire"), 1.0)

        proj = base_proj * factor * park_factor * ump_factor
        proys.append(proj)

        match = match_prop(row["pitcher"], lines_df)
        if match is not None:
            prop, score = match
            line = prop["line"]
            odds = prop["odds"]
            var_so = logs["SO"].var(ddof=1) if not logs.empty else None
            if var_so and var_so > proj:
                r = proj**2 / (var_so - proj)
                p = r / (r + proj)
                prob = 1 - nbinom.cdf(line - 1, r, p)
            else:
                prob = 1 - poisson.cdf(line - 1, proj)
            ev = (prob * odds) - (1 - prob)
            lines.append(line)
            odds_list.append(odds)
            probs.append(prob)
            evs.append(ev)
            books.append(prop["book"])
            if ev > 0.6 or ev < -0.3:
                print(f"⚠️ EV atípico para {row['FullName']} ({row['team']}): {ev:.3f} - Revisar datos")
            if ev > 0.3:
                ev_flag = "outlier"
            elif ev > 0.12:
                ev_flag = "ok"
            elif ev > 0.05:
                ev_flag = "warning"
            elif ev > -0.05:
                ev_flag = "riesgo"
            else:
                ev_flag = "no value"
            ev_validado.append(ev_flag)
            if ev > 0.05:
                suggestions.append("Apostar")
            elif ev > -0.05:
                suggestions.append("Riesgo alto")
            else:
                suggestions.append("No apostar")
            match_score = score
            if not logs.empty and line is not None:
                streak = streak_SO_over(logs, line)
            else:
                streak = None
            streaks.append(streak)
        else:
            lines.append(None)
            odds_list.append(None)
            probs.append(None)
            evs.append(None)
            books.append(None)
            suggestions.append("Sin linea")
            match_score = 50
            ev_validado.append(None)
            streaks.append(None)

        ump_factor = UMP_K_FACTORS.get(row.get("umpire"), 1.0)
        log_weight = min(len(logs) / 5, 1.0)
        match_weight = match_score / 100
        ump_weight = 1 - min(abs(ump_factor - 1), 0.08) / 0.08
        conf = round((log_weight + match_weight + ump_weight) / 3, 2)
        confidences.append(conf)

    df["Proyeccion_Ajustada"] = [round(p, 2) for p in proys]
    df["Linea"] = lines
    df["Cuota"] = odds_list
    df["Book"] = books
    df["Prob_O" ] = [round(p, 3) if p is not None else None for p in probs]
    df["EV"] = [round(e, 3) if e is not None else None for e in evs]
    df["Sugerencia"] = suggestions
    df["Confidence"] = confidences
    df["EV_validado"] = ev_validado
    df["Racha_SO"] = streaks
    df["IP_por_GS"] = (df["IP"] / df["GS"]).round(2)
    return df


def main(fecha: Optional[str] = None, output: Optional[str] = None):
    if not fecha:
        fecha = mlb_next_games_date_local()
        print(f"⚾ Usando la fecha relevante de juegos MLB: {fecha}")

    prob_df = get_probable_pitchers(fecha)
    if prob_df.empty:
        print("⚠️ No se encontraron abridores para la fecha dada.")
        return

    print("🧢 Abridores obtenidos:")
    print(prob_df)

    year = datetime.now().year
    stats_df = get_pitching_stats(year)

    lines_df = get_k_props()

    abridores_df = filtrar_stats(stats_df, prob_df)
    if abridores_df.empty:
        print("⚠️ No se encontraron estadísticas para los abridores.")
        print("Pitchers buscados:", prob_df["pitcher"].tolist())
        print("Pitchers con stats encontrados:", abridores_df.get("FullName", pd.Series()).tolist())
        return

    start = (datetime.strptime(fecha, "%Y-%m-%d") - timedelta(days=7)).strftime("%Y-%m-%d")
    end = (datetime.strptime(fecha, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")

    resultado = agregar_proyecciones(abridores_df, prob_df, lines_df, start, end)

    columnas = [
        "FullName",
        "team",
        "opponent",
        "W",
        "L",
        "ERA",
        "GS",
        "IP",
        "SO9",
        "IP_por_GS",
        "Proyeccion_Ajustada",
        "Linea",
        "Cuota",
        "Book",
        "Prob_O",
        "EV",
        "EV_validado",
        "Sugerencia",
        "Confidence",
        "Racha_SO",
    ]
    columnas = [c for c in columnas if c in resultado.columns]
    resultado = resultado[columnas]

    # Mejores picks del día
    try:
        picks = resultado.dropna(subset=["EV", "Linea"])
        picks = picks[picks["EV_validado"].isin(["ok", "outlier"])]
        picks = picks[picks["Confidence"] > 0.12]
        top3 = picks.sort_values("EV", ascending=False).head(3)
        print("\n🏆 Mejores picks del día (top 3 EV):")
        for _, row in top3.iterrows():
            print(
                f"  - {row['FullName']} ({row['team']} vs {row['opponent']}): Línea {row['Linea']} EV={row['EV']} Conf={row['Confidence']}"
            )
    except Exception as e:
        print(f"No se pudo mostrar picks top: {e}")

    if output:
        salida = output
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        timestamp = datetime.now().strftime("%H%M%S")
        salida = os.path.join(script_dir, f"abridores_{fecha}_{timestamp}.xlsx")
    resultado.to_excel(salida, index=False)
    print(f"✅ Archivo generado: {salida}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pitcher projections")
    parser.add_argument("--date", help="Fecha YYYY-MM-DD")
    parser.add_argument("--output", help="Ruta del Excel de salida")
    args = parser.parse_args()
    main(args.date, args.output)