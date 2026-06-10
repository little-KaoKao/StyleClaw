import * as React from "react"
import { cva, type VariantProps } from "class-variance-authority"
import { Slot } from "radix-ui"

import { cn } from "@/lib/utils"

const badgeVariants = cva(
  "inline-flex w-fit shrink-0 items-center justify-center gap-1 overflow-hidden rounded-none border-2 border-foreground px-2 py-0.5 text-xs font-bold uppercase tracking-wider whitespace-nowrap transition-[color,box-shadow] focus-visible:ring-2 focus-visible:ring-offset-2 focus-visible:ring-foreground [&>svg]:pointer-events-none [&>svg]:size-3",
  {
    variants: {
      variant: {
        default: "bg-bauhaus-red text-white",
        red: "bg-bauhaus-red text-white",
        blue: "bg-bauhaus-blue text-white",
        yellow: "bg-bauhaus-yellow text-foreground",
        secondary: "bg-muted text-foreground",
        destructive: "bg-bauhaus-red text-white",
        outline: "bg-white text-foreground",
        ghost: "border-transparent bg-transparent text-foreground",
        link: "border-transparent bg-transparent text-foreground underline-offset-4 [a&]:hover:underline",
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
