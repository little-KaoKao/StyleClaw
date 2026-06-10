import * as React from "react"
import { cva, type VariantProps } from "class-variance-authority"
import { Slot } from "radix-ui"

import { cn } from "@/lib/utils"

const badgeVariants = cva(
  "inline-flex w-fit shrink-0 items-center justify-center gap-1 overflow-hidden rounded-full px-3 py-1 text-xs font-bold whitespace-nowrap transition-all focus-visible:outline-none focus-visible:ring-4 focus-visible:ring-clay-accent/30 [&>svg]:pointer-events-none [&>svg]:size-3",
  {
    variants: {
      variant: {
        default: "bg-clay-accent/10 text-clay-accent",
        red: "bg-clay-accent/10 text-clay-accent",
        blue: "bg-clay-sky/10 text-clay-sky",
        yellow: "bg-clay-amber/15 text-clay-amber",
        secondary: "bg-clay-surface text-muted",
        destructive: "bg-clay-accent-alt/10 text-clay-accent-alt",
        outline: "bg-white text-foreground",
        ghost: "bg-transparent text-foreground",
        link: "bg-transparent text-clay-accent underline-offset-4 [a&]:hover:underline",
      },
    },
    defaultVariants: {
      variant: "default",
    },
  }
)

function Badge({
  className,
  variant = "default",
  asChild = false,
  ...props
}: React.ComponentProps<"span"> &
  VariantProps<typeof badgeVariants> & { asChild?: boolean }) {
  const Comp = asChild ? Slot.Root : "span"

  return (
    <Comp
      data-slot="badge"
      data-variant={variant}
      className={cn(badgeVariants({ variant }), className)}
      {...props}
    />
  )
}

export { Badge, badgeVariants }
