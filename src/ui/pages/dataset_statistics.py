"""Dataset statistics page for interactive data exploration."""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd
import plotly.express as px
import streamlit as st


def _truncate_text(value: Any, max_len: int = 140) -> str:
    if value is None:
        return ""
    text = str(value).replace("\n", " ").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


@st.cache_data(show_spinner=False)
def _cached_summary(_data, scope: str, state: Optional[str], city: Optional[str]):
    try:
        return _data.get_dataset_summary(scope=scope, state=state, city=city)
    except Exception:
        return {
            "n_businesses": 0,
            "n_users": 0,
            "n_items": 0,
            "n_interactions": 0,
            "density_pct": 0.0,
            "min_year": None,
            "max_year": None,
        }


@st.cache_data(show_spinner=False)
def _cached_state_dist(_data):
    try:
        return _data.get_state_distribution(limit=60)
    except Exception:
        return pd.DataFrame(columns=["state", "n_businesses", "avg_rating"])


@st.cache_data(show_spinner=False)
def _cached_default_focus_state(_data):
    try:
        return _data.get_default_focus_state()
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def _cached_city_dist(_data, scope: str, state: Optional[str]):
    try:
        return _data.get_top_cities(scope=scope, state=state, limit=200)
    except Exception:
        return pd.DataFrame(columns=["city", "state", "n_businesses", "avg_rating"])


@st.cache_data(show_spinner=False)
def _cached_rating_dist(_data, scope: str, state: Optional[str], city: Optional[str]):
    try:
        return _data.get_rating_distribution(scope=scope, state=state, city=city)
    except Exception:
        return pd.DataFrame(columns=["stars", "review_count"])


@st.cache_data(show_spinner=False)
def _cached_review_volume(
    _data,
    scope: str,
    state: Optional[str],
    city: Optional[str],
    rating_min: Optional[float],
    rating_max: Optional[float],
):
    try:
        return _data.get_review_volume_by_year(
            scope=scope,
            state=state,
            city=city,
            rating_min=rating_min,
            rating_max=rating_max,
        )
    except Exception:
        return pd.DataFrame(columns=["year", "review_count"])


@st.cache_data(show_spinner=False)
def _cached_categories(_data, scope: str, state: Optional[str], city: Optional[str]):
    try:
        return _data.get_top_categories(scope=scope, state=state, city=city, limit=20)
    except Exception:
        return pd.DataFrame(columns=["category", "n_businesses", "avg_rating"])


@st.cache_data(show_spinner=False)
def _cached_activity(_data, scope: str, state: Optional[str], city: Optional[str]):
    try:
        return _data.get_activity_distributions(scope=scope, state=state, city=city)
    except Exception:
        return (
            pd.DataFrame(columns=["cnt", "num_users"]),
            pd.DataFrame(columns=["cnt", "num_items"]),
        )


@st.cache_data(show_spinner=False)
def _cached_sample_businesses(_data, scope: str, state: Optional[str], city: Optional[str]):
    try:
        return _data.get_sample_business_rows(scope=scope, state=state, city=city, limit=12)
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def _cached_sample_reviews(
    _data,
    scope: str,
    state: Optional[str],
    city: Optional[str],
    year_min: Optional[int],
    year_max: Optional[int],
    rating_min: Optional[float],
    rating_max: Optional[float],
):
    try:
        return _data.get_sample_review_rows(
            scope=scope,
            state=state,
            city=city,
            year_min=year_min,
            year_max=year_max,
            rating_min=rating_min,
            rating_max=rating_max,
            limit=12,
        )
    except Exception:
        return pd.DataFrame()


def _plot_activity(df: pd.DataFrame, x_col: str, y_col: str, title: str, y_title: str):
    if df.empty:
        st.info(f"No data for {title.lower()}.")
        return

    fig = px.line(
        df,
        x=x_col,
        y=y_col,
        title=title,
        labels={x_col: "Interactions per entity", y_col: y_title},
        markers=True,
        log_x=True,
        log_y=True,
    )
    fig.update_layout(height=360)
    st.plotly_chart(fig, width="stretch")


def show() -> None:
    data = st.session_state.get("data")
    if not data:
        st.error("Data service not initialized.")
        return

    st.title("Dataset Statistics")
    st.caption(
        "Interactive exploration of source data."
    )
    scope = "global"

    state_df = _cached_state_dist(data)
    default_focus_state = _cached_default_focus_state(data) or "PA"
    state_options = ["All"]
    if not state_df.empty and "state" in state_df:
        state_options += [str(v) for v in state_df["state"].dropna().tolist()]

    default_state_index = (
        state_options.index(default_focus_state)
        if default_focus_state in state_options
        else 0
    )

    st.info(
        f"Default state filter is `{default_focus_state}` because the model trains on this state subset, "
        "and loading it is faster than the full dataset."
    )

    filter_col1, filter_col2, filter_col3 = st.columns(3)

    with filter_col1:
        selected_state_label = st.selectbox(
            "State",
            options=state_options,
            index=default_state_index,
        )
        selected_state = None if selected_state_label == "All" else selected_state_label

    city_df = _cached_city_dist(data, scope, selected_state)
    city_options = ["All"]
    if not city_df.empty and "city" in city_df:
        city_options += [str(v) for v in city_df["city"].dropna().tolist()]

    with filter_col2:
        selected_city_label = st.selectbox("City", options=city_options, index=0)
        selected_city = None if selected_city_label == "All" else selected_city_label

    with filter_col3:
        rating_range = st.slider("Review stars", 1.0, 5.0, (1.0, 5.0), 0.5)

    summary = _cached_summary(data, scope, selected_state, selected_city)
    kpi_cols = st.columns(6)
    kpi_cols[0].metric("Businesses", f"{int(summary.get('n_businesses', 0)):,}")
    kpi_cols[1].metric("Users", f"{int(summary.get('n_users', 0)):,}")
    kpi_cols[2].metric("Items", f"{int(summary.get('n_items', 0)):,}")
    kpi_cols[3].metric("Interactions", f"{int(summary.get('n_interactions', 0)):,}")
    kpi_cols[4].metric("Sparsity (positive)", f"{float(summary.get('density_pct', 0.0)):.4f}%")
    kpi_cols[5].metric(
        "Year span",
        (
            f"{int(summary['min_year'])}-{int(summary['max_year'])}"
            if summary.get("min_year") is not None and summary.get("max_year") is not None
            else "N/A"
        ),
    )

    review_volume_df = _cached_review_volume(
        data,
        scope,
        selected_state,
        selected_city,
        rating_range[0],
        rating_range[1],
    )
    rating_dist_df = _cached_rating_dist(data, scope, selected_state, selected_city)
    categories_df = _cached_categories(data, scope, selected_state, selected_city)
    top_cities_df = city_df.head(20)

    chart_col_left, chart_col_right = st.columns(2)
    with chart_col_left:
        if not rating_dist_df.empty:
            fig = px.bar(
                rating_dist_df,
                x="stars",
                y="review_count",
                title="Rating Distribution",
                labels={"stars": "Stars", "review_count": "Reviews"},
            )
            fig.update_layout(height=360)
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("No rating distribution data.")

    with chart_col_right:
        if not review_volume_df.empty:
            fig = px.line(
                review_volume_df,
                x="year",
                y="review_count",
                title="Review Volume Over Time",
                labels={"year": "Year", "review_count": "Reviews"},
                markers=True,
            )
            fig.update_layout(height=360)
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("No yearly volume data.")

    chart_col_left, chart_col_right = st.columns(2)
    with chart_col_left:
        if not top_cities_df.empty:
            fig = px.bar(
                top_cities_df.sort_values("n_businesses").tail(15),
                x="n_businesses",
                y="city",
                color="avg_rating",
                orientation="h",
                title="Top Cities by Business Count",
                labels={"n_businesses": "Businesses", "city": "City", "avg_rating": "Avg rating"},
            )
            fig.update_layout(height=420)
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("No city distribution data.")

    with chart_col_right:
        if not categories_df.empty:
            plot_df = categories_df.head(15).copy()
            category_col = (
                "category"
                if "category" in plot_df.columns
                else ("categories" if "categories" in plot_df.columns else None)
            )
            if category_col is None:
                st.info("No category distribution data.")
            else:
                plot_df["category_short"] = plot_df[category_col].map(
                    lambda x: _truncate_text(x, 45)
                )
                fig = px.bar(
                    plot_df.sort_values("n_businesses"),
                    x="n_businesses",
                    y="category_short",
                    color="avg_rating",
                    orientation="h",
                    title="Top Categories",
                    labels={
                        "n_businesses": "Businesses",
                        "category_short": "Category",
                        "avg_rating": "Avg rating",
                    },
                )
                fig.update_layout(height=420)
                st.plotly_chart(fig, width="stretch")
        else:
            st.info("No category distribution data.")

    st.subheader("Activity Distributions (log scale)")
    load_activity = st.toggle(
        "Load activity distributions",
        value=False,
        help="Loads additional aggregate queries; enable when you need these charts.",
    )
    if load_activity:
        user_activity_df, item_activity_df = _cached_activity(
            data, scope, selected_state, selected_city
        )
        activity_col1, activity_col2 = st.columns(2)
        with activity_col1:
            _plot_activity(
                user_activity_df,
                x_col="cnt",
                y_col="num_users",
                title="User Activity Distribution",
                y_title="Number of users",
            )
        with activity_col2:
            _plot_activity(
                item_activity_df,
                x_col="cnt",
                y_col="num_items",
                title="Item Popularity Distribution",
                y_title="Number of items",
            )
    else:
        st.caption("Activity charts are paused for faster initial page load.")

    st.subheader("Geographic View")
    if not state_df.empty:
        geo_df = state_df.copy()
        fig = px.choropleth(
            geo_df,
            locations="state",
            locationmode="USA-states",
            color="n_businesses",
            scope="usa",
            color_continuous_scale="Blues",
            title="Business Intensity by State (Global)",
            hover_data={"avg_rating": True, "n_businesses": True},
        )
        fig.update_layout(height=420)
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("No geographic data available.")

    st.subheader("Sample Rows")
    load_samples = st.toggle(
        "Load sample rows",
        value=False,
        help="Loads table preview queries; enable when you want raw row samples.",
    )
    if load_samples:
        sample_business_df = _cached_sample_businesses(
            data, scope, selected_state, selected_city
        )
        sample_review_df = _cached_sample_reviews(
            data,
            scope,
            selected_state,
            selected_city,
            int(review_volume_df["year"].min()) if not review_volume_df.empty else None,
            int(review_volume_df["year"].max()) if not review_volume_df.empty else None,
            rating_range[0],
            rating_range[1],
        )

        samples_col1, samples_col2 = st.columns(2)
        with samples_col1:
            st.markdown("**Businesses**")
            st.dataframe(sample_business_df, width="stretch", hide_index=True)

        with samples_col2:
            st.markdown("**Reviews**")
            if "text" in sample_review_df.columns:
                sample_review_df = sample_review_df.copy()
                sample_review_df["text"] = sample_review_df["text"].map(_truncate_text)
            st.dataframe(sample_review_df, width="stretch", hide_index=True)
    else:
        st.caption("Sample tables are paused for faster initial page load.")
