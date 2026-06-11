import * as React from "react"
import { cva, type VariantProps } from "class-variance-authority"
import { Slot } from "radix-ui"

import { cn } from "@/lib/utils"
import { CLAY_BUTTON, PRIMARY_GRADIENT } from "@/lib/clay"

const buttonVariants = cva(
  cn(
    // inline-flex + items-center + justify-center keeps text + icons
    // always vertically/horizontally centered inside the button.
    "inline-flex shrink-0 items-center justify-center gap-2 whitespace-nowrap",
    "font-bold tracking-wide",
    "outline-none focus-visible:outline-none focus-visible:ring-4 focus-visible:ring-clay-accent/30",
    "disabled:opacity-50 disabled:pointer-events-none",
    "[&_svg]:pointer-events-none [&_svg]:shrink-0 [&_svg]:h-5 [&_svg]:w-5"
  ),
  {
    variants: {
      variant: {
        default: `${PRIMARY_GRADIENT} text-white ${CLAY_BUTTON}`,
        primary: `${PRIMARY_GRADIENT} text-white ${CLAY_BUTTON}`,
        secondary: `bg-white text-foreground ${CLAY_BUTTON}`,
        soft: "rounded-2xl bg-clay-surface text-muted-foreground shadow-[5px_5px_12px_rgba(160,150,180,0.22),-5px_-5px_12px_rgba(255,255,255,0.9)] transition-all duration-200 hover:-translate-y-0.5 hover:text-clay-accent active:scale-[0.96] active:shadow-clayPressed",
        outline:
          "border-2 border-clay-accent/20 bg-transparent text-clay-accent rounded-2xl transition-all duration-200 hover:border-clay-accent hover:bg-clay-accent/5 hover:-translate-y-1 active:scale-[0.92]",
        ghost:
          "text-foreground rounded-2xl hover:bg-clay-accent/10 hover:text-clay-accent transition-all duration-200",
        destructive: `bg-gradient-to-br from-[#F472B6] to-[#DB2777] text-white ${CLAY_BUTTON}`,
        // legacy Bauhaus variant names mapped to clay equivalents
        red: `${PRIMARY_GRADIENT} text-white ${CLAY_BUTTON}`,
        blue: `bg-gradient-to-br from-sky-400 to-sky-600 text-white ${CLAY_BUTTON}`,
        yellow: `bg-gradient-to-br from-amber-400 to-amber-500 text-white ${CLAY_BUTTON}`,
        link: "rounded-2xl bg-transparent text-clay-accent underline-offset-4 transition-all duration-200 hover:underline",
      },
      size: {
        default: "h-12 px-7",
        xs: "h-9 gap-1 px-3 text-xs [&_svg]:h-3.5 [&_svg]:w-3.5",
        sm: "h-11 px-5 text-sm",
        lg: "h-16 px-8 text-lg",
        icon: "h-12 w-12",
        "icon-xs": "h-9 w-9 [&_svg]:h-3.5 [&_svg]:w-3.5",
        "icon-sm": "h-11 w-11",
        "icon-lg": "h-16 w-16",
      },
      shape: {
        square: "",
        pill: "rounded-full",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
      shape: "square",
    },
  }
)

function Button({
  className,
  variant = "default",
  size = "default",
  shape = "square",
  asChild = false,
  ...props
}: React.ComponentProps<"button"> &
  VariantProps<typeof buttonVariants> & {
    asChild?: boolean
  }) {
  const Comp = asChild ? Slot.Root : "button"

  return (
    <Comp
      data-slot="button"
      data-variant={variant}
      data-size={size}
      className={cn(buttonVariants({ variant, size, shape, className }))}
      {...props}
    />
  )
}

export { Button, buttonVariants }
