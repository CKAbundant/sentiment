"""Pydantic class to store trade info."""

from datetime import datetime
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, Field, computed_field, field_validator

from config.variables import PriceAction


class StockTrade(BaseModel):
    ticker: str = Field(description="Stock ticker to be traded")
    entry_datetime: datetime = Field(
        description="Datetime (New York time) when opening long position"
    )
    entry_action: PriceAction = Field(description="Buy, sell or wait")
    entry_lots: Decimal = Field(
        description="Number of lots to open new open position",
        default=Decimal("1"),
        ge=1,
    )
    entry_price: Decimal = Field(description="Price when opening long position", gt=0)
    exit_datetime: datetime | None = Field(
        description="Datetime (New York time) when exiting long position", default=None
    )
    exit_action: PriceAction | None = Field(
        description="Opposite of 'entry_action' or 'wait'", default=None
    )
    exit_lots: Decimal | None = Field(
        description="Number of lots to close open position", default=Decimal("0"), ge=0
    )
    exit_price: Decimal | None = Field(
        description="Price when exiting long position", default=None, gt=0
    )

    @computed_field(description="Number of days held for trade")
    def days_held(self) -> int | None:
        if self.exit_datetime is not None and self.entry_datetime is not None:
            days_held = self.exit_datetime - self.entry_datetime
            return days_held.days
        return

    @computed_field(description="Profit/loss when trade completed")
    def profit_loss(self) -> Decimal | None:
        if self.exit_price is not None and self.entry_price is not None:
            profit_loss = (
                self.exit_price - self.entry_price
                if self.entry_action == "buy"
                else self.entry_price - self.exit_price
            )
            return profit_loss
        return

    @computed_field(description="Percentage return of trade")
    def percent_ret(self) -> Decimal | None:
        if (
            self.exit_price is not None
            and self.entry_price is not None
            and self.profit_loss is not None
        ):
            percent_ret = self.profit_loss / self.entry_price
            return percent_ret.quantize(Decimal("1.000000"))
        return

    @computed_field(description="daily percentage return of trade")
    def daily_ret(self) -> Decimal | None:
        if self.percent_ret is not None and self.days_held != 0:
            daily_ret = (1 + self.percent_ret) ** (1 / Decimal(str(self.days_held))) - 1
            return daily_ret.quantize(Decimal("1.000000"))

        if self.percent_ret is not None and self.days_held == 0:
            # daily return = percent return if closed within same day
            return self.percent_ret
        return

    @computed_field(description="Whether trade is profitable")
    def win(self) -> int | None:
        if (pl := self.percent_ret) is not None:
            return int(pl > 0)
        return

    model_config = {"validate_assignment": True}

    @field_validator("exit_datetime")
    def validate_exit_datetime(
        cls, exit_datetime: datetime | None, info: dict[str, Any]
    ) -> datetime | None:
        # Get entry datetime from StockTrade object
        entry_datetime = info.data.get("entry_datetime")

        if exit_datetime is not None and entry_datetime is not None:
            if exit_datetime < entry_datetime:
                raise ValueError("Exit datetime must be after entry datetime!")
        return exit_datetime

    @field_validator("exit_action")
    def validate_exit_action(
        cls, exit_action: PriceAction | None, info: dict[str, Any]
    ) -> Decimal | None:
        # Get entry_action from StockTrade object
        entry_action = info.data.get("entry_action")

        if exit_action is not None and entry_action is not None:
            if entry_action == "wait" and exit_action != "wait":
                raise ValueError(
                    "Exit action must be 'wait' if entry action is 'wait'."
                )
            if entry_action != "wait" and entry_action == exit_action:
                raise ValueError(
                    "Entry action cannot be the same as exit action except when both are 'wait'."
                )
            if entry_action != "wait" and exit_action == "wait":
                raise ValueError(
                    "Exit action cannot be wait if entry action is not 'wait'."
                )

        return exit_action

    @field_validator("exit_lots")
    def validate_exit_lots(
        cls, exit_lots: Decimal | None, info: dict[str, Any]
    ) -> Decimal | None:
        # Get entry_lots from StockTrade object
        entry_lots = info.data.get("entry_lots")

        if exit_lots is not None and entry_lots is not None:
            if exit_lots > entry_lots:
                raise ValueError("Exit lots must be equal or less than entry lots.")

            if exit_lots < 0 or entry_lots < 0:
                raise ValueError(f"Entry lots and exit lots must be positive.")

        return exit_lots
