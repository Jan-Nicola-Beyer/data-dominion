"""Home frame — summary stats and quick-launch import button."""

from __future__ import annotations
from typing import TYPE_CHECKING

import customtkinter as ctk

import datalens_v3_opt.constants as C

if TYPE_CHECKING:
    from datalens_v3_opt.ui.app import App


class HomeFrame(ctk.CTkFrame):
    def __init__(self, master, app: App):
        super().__init__(master, fg_color=C.BG, corner_radius=0)
        self.app = app
        self._build()

    def _build(self):
        ctk.CTkLabel(self, text="Data Dominion",
                     font=("Segoe UI", 28, "bold"), text_color=C.TEXT
                     ).pack(pady=(60, 4))
        ctk.CTkLabel(self,
                     text="Unified Data Analysis System",
                     font=("Segoe UI", 14), text_color=C.MUTED
                     ).pack(pady=(0, 40))

        self._stat_frame = ctk.CTkFrame(self, fg_color="transparent")
        self._stat_frame.pack()
        self._stat_cards: dict = {}  # label -> (card, value_label)

        ctk.CTkButton(self, text="Import Dataset", width=200, height=44,
                      fg_color=C.ACCENT, font=("Segoe UI", 13, "bold"),
                      command=self.app.open_import_wizard
                      ).pack(pady=(40, 8))
        ctk.CTkLabel(self,
                     text="Import multiple datasets, then merge them in the Datasets panel.",
                     text_color=C.MUTED, font=("Segoe UI", 11)
                     ).pack()

    def refresh(self):
        stats = [
            ("Datasets",    str(len(self.app.dm.entries))),
            ("Active Rows", str(len(self.app.df))),
            ("Columns",     str(len(self.app.df.columns))),
        ]
        for label, val in stats:
            if label in self._stat_cards:
                # Update existing card value
                self._stat_cards[label].configure(text=val)
            else:
                # Create card once
                card = ctk.CTkFrame(self._stat_frame, fg_color=C.CARD,
                                    corner_radius=10, width=130)
                card.pack(side="left", padx=10)
                card.pack_propagate(False)
                val_lbl = ctk.CTkLabel(card, text=val,
                                       font=("Segoe UI", 22, "bold"),
                                       text_color=C.ACCENT)
                val_lbl.pack(pady=(10, 0))
                ctk.CTkLabel(card, text=label, font=("Segoe UI", 11),
                             text_color=C.MUTED).pack(pady=(0, 10))
                self._stat_cards[label] = val_lbl

    def rebuild(self):
        for w in self.winfo_children():
            w.destroy()
        self.configure(fg_color=C.BG)
        self._stat_cards = {}
        self._build()
        self.refresh()
