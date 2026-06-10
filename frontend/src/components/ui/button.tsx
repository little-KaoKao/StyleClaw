import * as React from "react"
import { cva, type VariantProps } from "class-variance-authority"
import { Slot } from "radix-ui"

import { cn } from "@/lib/utils"
import { SHADOW_SM, PRESS } from "@/lib/bauhaus"

const buttonVariants = cva(
  cn(
    // CRITICAL: inline-flex + items-center + justify-center keeps text + icons
    // always vertically/horizontally centered inside the button.
    "inline-flex shrink-0 items-center justify-center gap-2 whitespace-nowrap",
    "border-2 border-foreground rounded-none font-bold uppercase tracking-wider",
    "outline-none focus-visible:ring-2 focus-visible:ring-offset-2 focus-visible:ring-foreground",
    "disabled:opacity-50 disabled:pointer-events-none",
    "[&_svg]:pointer-events-none [&_svg]:shrink-0 [&_svg]:h-5 [&_svg]:w-5",
    SHADOW_SM,
    PRESS,
    "hover:opacity-90"
  ),
  {
    variants: {
      variant: {
        default: "bg-bauhaus-red text-white",
        red: "bg-bauhaus-red text-white",
        blue: "bg-bauhaus-blue text-white",
        yellow: "bg-bauhaus-yellow text-foreground",
        outline: "bg-white text-foreground",
        // ghost drops the border + shadow + press effect
        ghost:
          "border-transparent shadow-none hover:bg-muted hover:opacity-100 active:translate-x-0 active:translate-y-0",
        // map legacy variants to Bauhaus equivalents so old call sites don't break
        destructive: "bg-bauhaus-red text-white",
        secondary: "bg-white text-foreground",
        link: "border-transparent shadow-none bg-transparent text-foreground underline-offset-4 hover:underline hover:opacity-100 active:translate-x-0 active:translate-y-0",
      },
      size: {
        default: "h-10 px-4 text-sm",
        xs: "h-7 gap-1 px-2 text-xs [&_svg]:h-3.5 [&_svg]:w-3.5",
        sm: "h-8 gap-1.5 px-3 text-xs",
        lg: "h-12 px-6 text-base",
        icon: "size-10",
        "icon-xs": "size-7 [&_svg]:h-3.5 [&_svg]:w-3.5",
        "icon-sm": "size-8",
        "icon-lg": "size-12",
      },
      shape: {
        square: "rounded-none",
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
