# from reportlab.lib.pagesizes import letter
# from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem
# from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
# from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
# from reportlab.lib import colors

# def create_pdf(filename):
#     doc = SimpleDocTemplate(
#         filename,
#         pagesize=letter,
#         rightMargin=72,
#         leftMargin=72,
#         topMargin=72,
#         bottomMargin=72
#     )

#     story = []
#     styles = getSampleStyleSheet()

#     # --- Custom Styles for a "Professional Report" Look ---
    
#     # Title Style
#     title_style = ParagraphStyle(
#         'CustomTitle',
#         parent=styles['Heading1'],
#         fontName='Times-Bold',
#         fontSize=18,
#         spaceAfter=12,
#         alignment=TA_LEFT,
#         textColor=colors.black
#     )

#     # Sub-Title / Author Style
#     author_style = ParagraphStyle(
#         'CustomAuthor',
#         parent=styles['Normal'],
#         fontName='Times-Italic',
#         fontSize=12,
#         spaceAfter=24,
#         textColor=colors.darkgrey
#     )

#     # Section Header Style
#     h2_style = ParagraphStyle(
#         'CustomH2',
#         parent=styles['Heading2'],
#         fontName='Times-Bold',
#         fontSize=14,
#         spaceBefore=12,
#         spaceAfter=6,
#         textColor=colors.black
#     )

#     # Body Text Style (Serif font looks more academic/human)
#     body_style = ParagraphStyle(
#         'CustomBody',
#         parent=styles['Normal'],
#         fontName='Times-Roman',
#         fontSize=11,
#         leading=14,  # Line spacing
#         alignment=TA_JUSTIFY,
#         spaceAfter=6
#     )

#     # Bullet Point Style
#     bullet_style = ParagraphStyle(
#         'CustomBullet',
#         parent=body_style,
#         firstLineIndent=0,
#         leftIndent=0,
#         spaceAfter=3
#     )

#     # --- Content Construction ---

#     # 1. Header
#     story.append(Paragraph("Project: Mean Reversion Calendar Spread Strategy", title_style))
#     story.append(Paragraph("Author: Madhav Gupta", author_style))

#     # 2. Strategy Overview
#     story.append(Paragraph("1. Strategy Overview", h2_style))
#     text_overview = (
#         "This project implements a statistical arbitrage strategy on NSE Futures Calendar Spreads "
#         "(Near Month vs. Far Month). The hypothesis is that the spread between these two contracts "
#         "is mean-reverting. We utilize a Z-Score based trigger to identify overextended spreads "
#         "and capture the reversion to the mean."
#     )
#     story.append(Paragraph(text_overview, body_style))
#     story.append(Spacer(1, 6))

#     # 3. Tradable Spread Identification
#     story.append(Paragraph("2. Tradable Spread Identification", h2_style))
#     text_id_intro = (
#         "Analysis of initial backtests revealed that many pairs suffered from excessive slippage and "
#         "transaction costs, turning gross profits into net losses (e.g., LODHA). To \"identify tradable spreads,\" "
#         "we implicitly filter for quality by:"
#     )
#     story.append(Paragraph(text_id_intro, body_style))

#     # List for Section 2
#     items_id = [
#         ListItem(Paragraph("<b>Liquidity Thresholds:</b> Stocks with zero volume or insufficient data points to calculate Z-scores are automatically excluded.", bullet_style)),
#         ListItem(Paragraph("<b>High Conviction Entry:</b> We increased the Z-Score Entry Threshold to 2.0 (from 1.5). This acts as a quality filter, ensuring we only trade when the spread deviation is statistically significant, thereby improving the signal-to-noise ratio and reducing the impact of fixed transaction costs.", bullet_style))
#     ]
#     story.append(ListFlowable(items_id, bulletType='bullet', start='circle', leftIndent=20))
#     story.append(Spacer(1, 6))

#     # 4. Trading Logic
#     story.append(Paragraph("3. Trading Logic", h2_style))
    
#     # List for Section 3
#     items_logic = [
#         ListItem(Paragraph("<b>Signal:</b> 60-day Rolling Z-Score of the Spread.", bullet_style)),
#         ListItem(Paragraph("<b>Entry:</b> |Z-Score| > 2.0.", bullet_style)),
#         ListItem(ListFlowable([
#             ListItem(Paragraph("Short Spread (Sell Near / Buy Far) if Z > 2.0.", bullet_style)),
#             ListItem(Paragraph("Long Spread (Buy Near / Sell Far) if Z < -2.0.", bullet_style))
#         ], bulletType='bullet', start='square', leftIndent=20)),
#         ListItem(Paragraph("<b>Exit:</b>", bullet_style)),
#         ListItem(ListFlowable([
#             ListItem(Paragraph("Take Profit (TP): 0.5 Sigma relative to entry.", bullet_style)),
#             ListItem(Paragraph("Stop Loss (SL): 1.5 Sigma relative to entry (Tightened to protect capital).", bullet_style))
#         ], bulletType='bullet', start='square', leftIndent=20)),
#         ListItem(Paragraph("<b>Expiry:</b> Force close all positions at End-Of-Day on Near Month Expiry.", bullet_style))
#     ]
#     story.append(ListFlowable(items_logic, bulletType='bullet', start='circle', leftIndent=20))
#     story.append(Spacer(1, 6))

#     # 5. Metrics & Assumptions
#     story.append(Paragraph("4. Metrics & Assumptions", h2_style))
#     items_metrics = [
#         ListItem(Paragraph("<b>Max Gross Qty:</b> Calculated as the maximum absolute open position (lots) held at any minute.", bullet_style)),
#         ListItem(Paragraph("<b>Max Delta Qty:</b> Calculated as the maximum change in lots in a single minute (trade size).", bullet_style)),
#         ListItem(Paragraph("<b>Costs:</b> Slippage is calculated based on the observed Bid-Ask spread at the moment of execution. Commission is modelled at ~2 bps per leg.", bullet_style))
#     ]
#     story.append(ListFlowable(items_metrics, bulletType='bullet', start='circle', leftIndent=20))
#     story.append(Spacer(1, 6))

#     # 6. Execution
#     story.append(Paragraph("5. Execution", h2_style))
#     text_exec = (
#         "The simulation is powered by a vectorized Python engine (<i>simulation_engine.py</i>) optimized "
#         "for speed using <i>ProcessPoolExecutor</i>. The driver script (<i>problem2_runner.py</i>) aggregates these "
#         "results and generates the final leaderboard."
#     )
#     story.append(Paragraph(text_exec, body_style))

#     # Build PDF
#     doc.build(story)
#     print(f"Success! Generated: {filename}")

# if __name__ == "__main__":
#     create_pdf("Problem2_readme.pdf")


# from reportlab.lib.pagesizes import letter
# from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem
# from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
# from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
# from reportlab.lib import colors

# def create_pdf(filename):
#     doc = SimpleDocTemplate(
#         filename,
#         pagesize=letter,
#         rightMargin=72,
#         leftMargin=72,
#         topMargin=72,
#         bottomMargin=72
#     )

#     story = []
#     styles = getSampleStyleSheet()

#     # --- Custom Styles ---
#     title_style = ParagraphStyle(
#         'CustomTitle',
#         parent=styles['Heading1'],
#         fontName='Times-Bold',
#         fontSize=18,
#         spaceAfter=12,
#         alignment=TA_LEFT,
#         textColor=colors.black
#     )

#     author_style = ParagraphStyle(
#         'CustomAuthor',
#         parent=styles['Normal'],
#         fontName='Times-Italic',
#         fontSize=12,
#         spaceAfter=24,
#         textColor=colors.darkgrey
#     )

#     h2_style = ParagraphStyle(
#         'CustomH2',
#         parent=styles['Heading2'],
#         fontName='Times-Bold',
#         fontSize=14,
#         spaceBefore=12,
#         spaceAfter=6,
#         textColor=colors.black
#     )

#     body_style = ParagraphStyle(
#         'CustomBody',
#         parent=styles['Normal'],
#         fontName='Times-Roman',
#         fontSize=11,
#         leading=14,
#         alignment=TA_JUSTIFY,
#         spaceAfter=6
#     )

#     bullet_style = ParagraphStyle(
#         'CustomBullet',
#         parent=body_style,
#         firstLineIndent=0,
#         leftIndent=0,
#         spaceAfter=3
#     )

#     # --- Content Construction ---

#     # 1. Header
#     story.append(Paragraph("Project: Mean Reversion Calendar Spread Strategy", title_style))
#     story.append(Paragraph("Author: Madhav Gupta", author_style))

#     # 2. Strategy Overview
#     story.append(Paragraph("1. Strategy Overview", h2_style))
#     text_overview = (
#         "This project implements a statistical arbitrage strategy on NSE Futures Calendar Spreads "
#         "(Near Month vs. Far Month). The hypothesis is that the spread between these two contracts "
#         "is mean-reverting. We utilize a Z-Score based trigger to identify overextended spreads "
#         "and capture the reversion to the mean."
#     )
#     story.append(Paragraph(text_overview, body_style))
#     story.append(Spacer(1, 6))

#     # 3. Tradable Spread Identification (Including Grid Search)
#     story.append(Paragraph("2. Tradable Spread Identification & Optimization", h2_style))
#     text_id_intro = (
#         "Analysis of initial backtests revealed that many pairs suffered from excessive slippage and "
#         "transaction costs, turning gross profits into net losses (e.g., LODHA). To \"identify tradable spreads,\" "
#         "we implicitly filter for quality by:"
#     )
#     story.append(Paragraph(text_id_intro, body_style))

#     # List for Section 2
#     items_id = [
#         ListItem(Paragraph("<b>Liquidity Thresholds:</b> Stocks with zero volume or insufficient data points to calculate Z-scores are automatically excluded.", bullet_style)),
#         ListItem(Paragraph("<b>Grid Search Optimization:</b> We utilized a <b>Grid Search</b> approach (running <i>run_grid.py</i>) to simulate various combinations of Entry Thresholds (1.5, 2.0, 2.5) and Stop Losses. The analysis confirmed that higher entry thresholds yielded better risk-adjusted returns.", bullet_style)),
#         ListItem(Paragraph("<b>High Conviction Entry:</b> Based on the Grid Search results, we increased the Z-Score Entry Threshold to <b>2.0</b> (from 1.5). This acts as a quality filter, ensuring we only trade when the spread deviation is statistically significant, thereby improving the signal-to-noise ratio and reducing the impact of fixed transaction costs.", bullet_style))
#     ]
#     story.append(ListFlowable(items_id, bulletType='bullet', start='circle', leftIndent=20))
#     story.append(Spacer(1, 6))

#     # 4. Trading Logic
#     story.append(Paragraph("3. Trading Logic", h2_style))
    
#     # List for Section 3
#     items_logic = [
#         ListItem(Paragraph("<b>Signal:</b> 60-day Rolling Z-Score of the Spread.", bullet_style)),
#         ListItem(Paragraph("<b>Entry:</b> |Z-Score| > 2.0.", bullet_style)),
#         ListItem(ListFlowable([
#             ListItem(Paragraph("Short Spread (Sell Near / Buy Far) if Z > 2.0.", bullet_style)),
#             ListItem(Paragraph("Long Spread (Buy Near / Sell Far) if Z < -2.0.", bullet_style))
#         ], bulletType='bullet', start='square', leftIndent=20)),
#         ListItem(Paragraph("<b>Exit:</b>", bullet_style)),
#         ListItem(ListFlowable([
#             ListItem(Paragraph("Take Profit (TP): 0.5 Sigma relative to entry.", bullet_style)),
#             ListItem(Paragraph("Stop Loss (SL): 1.5 Sigma relative to entry (Tightened to protect capital).", bullet_style))
#         ], bulletType='bullet', start='square', leftIndent=20)),
#         ListItem(Paragraph("<b>Expiry:</b> Force close all positions at End-Of-Day on Near Month Expiry.", bullet_style))
#     ]
#     story.append(ListFlowable(items_logic, bulletType='bullet', start='circle', leftIndent=20))
#     story.append(Spacer(1, 6))

#     # 5. Metrics & Assumptions
#     story.append(Paragraph("4. Metrics & Assumptions", h2_style))
#     items_metrics = [
#         ListItem(Paragraph("<b>Max Gross Qty:</b> Calculated as the maximum absolute open position (lots) held at any minute.", bullet_style)),
#         ListItem(Paragraph("<b>Max Delta Qty:</b> Calculated as the maximum change in lots in a single minute (trade size).", bullet_style)),
#         ListItem(Paragraph("<b>Costs:</b> Slippage is calculated based on the observed Bid-Ask spread at the moment of execution. Commission is modelled at ~2 bps per leg.", bullet_style))
#     ]
#     story.append(ListFlowable(items_metrics, bulletType='bullet', start='circle', leftIndent=20))
#     story.append(Spacer(1, 6))

#     # 6. Execution
#     story.append(Paragraph("5. Execution", h2_style))
#     text_exec = (
#         "The simulation is powered by a vectorized Python engine (<i>simulation_engine.py</i>) optimized "
#         "for speed using <i>ProcessPoolExecutor</i>. The driver script (<i>run_sim.py</i>) aggregates these "
#         "results and generates the final leaderboard."
#     )
#     story.append(Paragraph(text_exec, body_style))

#     # Build PDF
#     doc.build(story)
#     print(f"Success! Generated: {filename}")

# if __name__ == "__main__":
#     create_pdf("Problem2_readme.pdf")

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
from reportlab.lib import colors

def create_pdf(filename):
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72
    )

    story = []
    styles = getSampleStyleSheet()

    # --- Custom Styles ---
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontName='Times-Bold',
        fontSize=18,
        spaceAfter=12,
        alignment=TA_LEFT,
        textColor=colors.black
    )

    author_style = ParagraphStyle(
        'CustomAuthor',
        parent=styles['Normal'],
        fontName='Times-Italic',
        fontSize=12,
        spaceAfter=24,
        textColor=colors.darkgrey
    )

    h2_style = ParagraphStyle(
        'CustomH2',
        parent=styles['Heading2'],
        fontName='Times-Bold',
        fontSize=14,
        spaceBefore=12,
        spaceAfter=6,
        textColor=colors.black
    )

    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontName='Times-Roman',
        fontSize=11,
        leading=14,
        alignment=TA_JUSTIFY,
        spaceAfter=6
    )

    bullet_style = ParagraphStyle(
        'CustomBullet',
        parent=body_style,
        firstLineIndent=0,
        leftIndent=0,
        spaceAfter=3
    )

    # --- Content Construction ---

    # 1. Header
    story.append(Paragraph("Project: Mean Reversion Calendar Spread Strategy", title_style))
    story.append(Paragraph("Author: Madhav Gupta", author_style))

    # 2. Strategy Overview
    story.append(Paragraph("1. Strategy Overview", h2_style))
    text_overview = (
        "This project implements a statistical arbitrage strategy on NSE Futures Calendar Spreads "
        "(Near Month vs. Far Month). The hypothesis is that the spread between these two contracts "
        "is mean-reverting. We utilize a Z-Score based trigger to identify overextended spreads "
        "and capture the reversion to the mean."
    )
    story.append(Paragraph(text_overview, body_style))
    story.append(Spacer(1, 6))

    # 3. Tradable Spread Identification (With Grid Search Analysis)
    story.append(Paragraph("2. Tradable Spread Identification & Optimization", h2_style))
    text_id_intro = (
        "To identify the most robust tradable spreads and minimize slippage impact, we conducted "
        "a rigorous analysis combining liquidity filtering with a parameter grid search."
    )
    story.append(Paragraph(text_id_intro, body_style))

    # List for Section 2
    items_id = [
        ListItem(Paragraph("<b>Liquidity Filtering:</b> Stocks with zero volume or insufficient history for Z-score calculation were automatically excluded to prevent execution errors.", bullet_style)),
        ListItem(Paragraph("<b>Grid Search Optimization:</b> We utilized a <b>Grid Search</b> (<i>run_grid.py</i>) to simulate combinations of Entry Thresholds (1.5, 2.0, 2.5), Take Profits (0.5, 1.0), and Stop Losses. The results were aggregated into a leaderboard.", bullet_style)),
        ListItem(Paragraph("<b>Parameter Selection:</b> The leaderboard revealed that lower Entry thresholds (1.5) and tight Take Profits (0.5) consistently yielded negative Net PnL due to churn costs. The optimal configuration was identified as <b>Entry=2.5, TP=1.0</b>, which generated the highest Net PnL (~5.3L) by trading less frequently but with higher conviction.", bullet_style))
    ]
    story.append(ListFlowable(items_id, bulletType='bullet', start='circle', leftIndent=20))
    story.append(Spacer(1, 6))

    # 4. Trading Logic
    story.append(Paragraph("3. Trading Logic", h2_style))
    
    # List for Section 3
    items_logic = [
        ListItem(Paragraph("<b>Signal:</b> 60-day Rolling Winsorized Z-Score of the Spread.", bullet_style)),
        ListItem(Paragraph("<b>Entry:</b> |Z-Score| > 2.5 (Selected via Grid Search).", bullet_style)),
        ListItem(ListFlowable([
            ListItem(Paragraph("Short Spread (Sell Near / Buy Far) if Z > 2.5.", bullet_style)),
            ListItem(Paragraph("Long Spread (Buy Near / Sell Far) if Z < -2.5.", bullet_style))
        ], bulletType='bullet', start='square', leftIndent=20)),
        ListItem(Paragraph("<b>Exit:</b>", bullet_style)),
        ListItem(ListFlowable([
            ListItem(Paragraph("Take Profit (TP): 1.0 Sigma relative to entry (Allowed profits to run).", bullet_style)),
            ListItem(Paragraph("Stop Loss (SL): 2.0 Sigma relative to entry.", bullet_style))
        ], bulletType='bullet', start='square', leftIndent=20)),
        ListItem(Paragraph("<b>Expiry:</b> Force close all positions at End-Of-Day on Near Month Expiry.", bullet_style))
    ]
    story.append(ListFlowable(items_logic, bulletType='bullet', start='circle', leftIndent=20))
    story.append(Spacer(1, 6))

    # 5. Metrics & Assumptions
    story.append(Paragraph("4. Metrics & Assumptions", h2_style))
    items_metrics = [
        ListItem(Paragraph("<b>Max Gross Qty:</b> Calculated as the maximum absolute open position (lots) held at any minute.", bullet_style)),
        ListItem(Paragraph("<b>Max Delta Qty:</b> Calculated as the maximum change in lots in a single minute (trade size).", bullet_style)),
        ListItem(Paragraph("<b>Costs:</b> Slippage is calculated based on the observed Bid-Ask spread at the moment of execution. Commission is modelled at ~2 bps per leg.", bullet_style))
    ]
    story.append(ListFlowable(items_metrics, bulletType='bullet', start='circle', leftIndent=20))
    story.append(Spacer(1, 6))

    # 6. Execution
    story.append(Paragraph("5. Execution", h2_style))
    text_exec = (
        "The simulation is powered by a vectorized Python engine (<i>simulation_engine.py</i>) optimized "
        "for speed using <i>ProcessPoolExecutor</i>. The driver script (<i>problem2_runner.py</i>) runs the optimal "
        "parameter set and generates the final <i>Results.csv</i>."
    )
    story.append(Paragraph(text_exec, body_style))

    # Build PDF
    doc.build(story)
    print(f"Success! Generated: {filename}")

if __name__ == "__main__":
    create_pdf("Problem2_readme.pdf")